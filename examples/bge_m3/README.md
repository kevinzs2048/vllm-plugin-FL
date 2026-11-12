# BGE M3 优化代码与鲲鹏 950 测试说明

更新日期：2026 年 11 月 12 日。目标机器：鲲鹏 950，192 核，4 个 NUMA 节点。

本次交接包含已有 BGE-M3 优化的源码、组件分工和新服务器的测试步骤。已有实测平台是 M5 Pro 与 CIX P1；鲲鹏 950 的指令支持、最优线程数、NUMA 扩展和性能均待现场确认。本分支用于移交测试，不是社区已经合入的发布版本。按用户要求，本次没有继续扩大旧平台测试。

**先跑通单 NUMA，再测线程扩展，最后测四 NUMA 的独立实例吞吐。不要直接把单次前向设置成 192 线程。** 当前插件的阶段线程策略上限为 64；若每 NUMA 确有 48 个可用物理核，可先覆盖 8、16、24、32、48 线程。192 若包含 SMT 线程，应先按每物理核一个逻辑 CPU 测试。

## 1 模型和量化合同

BGE-M3 是编码器，24 层，hidden size 1024，16 个注意力头，每头 64 维。本文测试输入文本到 1024 维 dense CLS 向量并归一化，最大长度 8192 token。没有逐 token 生成，不能使用 TG128 衡量。sparse、ColBERT、Milvus 索引与检索不属于这里已验证的编码吞吐。

| 项目 | 当前实现 |
| --- | --- |
| 原始模型 | HF BAAI/bge-m3，原文件标注 FP32，运行时加载 BF16 |
| 原始 revision | `5617a9f61b028005a4858fdac845db406aefb181` |
| 主体 W8 | 对称 INT8，每输出通道 scale |
| 输入 A8 | 每次前向动态逐 token 量化，不是静态校准激活 |
| 校准 | 选定路径使用外置 `sq-fit-scales.pt`，必须匹配模型文件 |
| 非主体线性权重 | Embedding、norm、bias 和小 head 保留 BF16 |
| G32 | **本路径不是 W8A8 G32**。内部布局要求 K 按 32 对齐，不等于分组量化 |
| LM Head | BGE 没有生成式大词表 LM Head，不能套用 MiniCPM 的输出头说明 |

完整保留模型仓库最稳妥。最少文件为：`config.json`、`pytorch_model.bin`、`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`、`sentencepiece.bpe.model`、`colbert_linear.pt`、`sparse_linear.pt`。后两个 head 即使 dense 测试也会被模型类加载。

当前 W8 是启动时转换、准备打包布局，不是已发布的独立 INT8 checkpoint。量化生成脚本、模型、校准数据不提交到这三个功能仓库；校准数据单独随测试材料交付并记录 SHA256，不能只换模型文件继续套用旧尺度。

## 2 组件分工与代码位置

```text
应用文本或固定 token IDs
  → vLLM 0.24 CPU pooling runner 和调度器
  → vLLM-Plugin-FL：模型注册、权重准备生命周期、模型实例算子选择、线程策略
  → FlagGems：prepared 权重、W8A8 GEMM、FFN、GELU、残差 LayerNorm
  → FlagTree-CPU：Triton CPU 编译与 TLE Raw 调用能力
  → 已缓存的编译入口、FlagGems 原生叶函数、Torch 使用的同一 OpenMP runtime
  → 当前 CPU 实际支持的指令
```

| 仓库分支 | 负责内容 | 主要文件 |
| --- | --- | --- |
| [FlagGems bge-m3](https://github.com/kevinzs2048/FlagGems/tree/bge-m3) | 算术、打包、融合与原生叶函数构建 | `src/flag_gems/runtime/backend/_arm/fused/prepared_w8a8_*.py`；`_arm/tle_raw/csrc/w8a8_i8mm.cpp`、`w8a8_sme.cpp`、`w8a8_add_norm.cpp`、`elementwise.cpp`、`attention_sme.cpp` |
| [vLLM-Plugin-FL bge-m3](https://github.com/kevinzs2048/vllm-plugin-FL/tree/bge-m3) | 模型接入、平台路径与请求线程策略 | `vllm_fl/models/bge_m3_arm.py`、`bge_m3_quant.py`；`worker/cpu_thread_policy.py`、`cpu_phase_worker.py`；`csrc/cpu/encoder_attention.cpp` |
| [FlagTree-CPU bge-m3](https://github.com/kevinzs2048/flagtree-cpu/tree/bge-m3) | 既有 CPU Raw、ABI 和链接缓存能力 | `third_party/cpu` 与 CPU `tle_ops` 相关提交；本轮没有新增 BGE 专用编译器算法 |

FlagTree 分支继承 `cpu-tle-raw` 的四个功能提交，HEAD 为 `34f4c115251ebe1b68736a457fd0021c661fbaa1`。它是编译器依赖基线；不能把旧预编译二进制的测试写成这个新社区基线上整编译器重建已通过。其余源码以测试时记录的 git commit 为准。

**没有为这次整合修改 vLLM 核心仓库。** 插件仍依赖 vLLM 0.24 的 worker、模型类和 CPU attention 接口，不能推断任意 vLLM 版本可用。M5 旧运行包包含 macOS 构建集成，也不能称为未改社区 wheel。Linux 不需要 Mac Runtime 启动器；需要对应 Linux CPU 软件环境。

## 3 具体做过哪些优化

| 优化 | 改了什么 | 生效条件与代价 |
| --- | --- | --- |
| 权重准备复用 | 启动时完成平滑、W8 转换与重排，前向复用 prepared plan | 降低重复转换；启动慢、准备期间有临时内存；模型生命周期与 state_dict 重载仍需单独验证 |
| I8MM GEMM | 使用 `vmmlaq_s32` 对应的 Arm INT8 矩阵指令，融合 A8 准备、scale、bias、BF16 写回；按形状选择遍历顺序 | 需要实际 FEAT_I8MM。不是把 BF16 直接称为 I8MM 加速 |
| SME 与 Neon 分工 | M5 将互不重叠的输出块交给 SME 的 INT8 外积和 Neon I8MM 工作线程 | 是 M5 策略，不直接搬到鲲鹏；M5 两种布局各 288 MiB，存在空间成本 |
| 残差与 LayerNorm 融合 | 将线性输出、残差相加和 norm 的相关处理融合，减少独立调用、中间数据往返与并行区 | 保留 BF16 舍入顺序；GCC/Clang 的 FMA 编译选项不能随意统一 |
| 短输入 FFN 融合 | 将上投影、GELU、下投影与输出 norm 合并组织；P1 符合形状的路径复用 OpenMP 团队 | 主要覆盖总 scheduled rows ≤128；多 batch 按总行数判断。M5 与 P1 的团队组织并非完全相同 |
| GELU 查表 | 利用 BF16 只有 65536 种位模式的特点，准备完整映射；M5 长输入使用 Triton 双表保留向量和尾部舍入 | 查表是运行时算术，表准备另算。M5 双表 256 KiB；P1 未采用较慢的长输入 Triton 候选 |
| 短输入执行图 | 在模型实例上编译短输入 encoder，保留动态形状、请求边界和 Dynamo 检查 | 只在合格的单序列路径启用；多请求或超出范围回到原路径。没有恢复用户否决的模型结构检查缓存 |
| 线程等待策略 | 连续小算子间保持适当短暂等待，减少立即睡眠后反复唤醒 | GNU `GOMP_SPINCOUNT` 是自旋次数，不是毫秒；LLVM `KMP_BLOCKTIME` 是另一套机制。忙等消耗 CPU，低频请求须独立测试 |
| 按输入规模配线程 | 小矩阵减少团队调度负担，大矩阵再扩大并行度；Torch、OpenMP、Raw GEMM 的线程要一起核对 | BGE 是编码器，不应照搬生成模型的 prefill/decode 两套成绩。192 核服务器需要重新找拐点 |

TLE Raw **允许 C++/intrinsics 叶函数**。实际路径是 Triton CPU 编译入口调用 FlagGems 的原生叶函数，不等于所有算术都是纯 Triton DSL。当前还有原生 vLLM/PyTorch 的注意力、embedding 和 pooling。`encoder_attention.cpp` 桥接仍调用 vLLM 数学实现，不能统计成 FlagGems 自有 attention。

当前缓存了 Triton 编译产物和函数地址，但不能因为安装包包含 `libtriton_jit` 就声称 BGE 每条热路径都通过该动态库调用。

## 4 新服务器先确认硬件和环境

先保存以下信息，不改系统设置：

```bash
mkdir -p "$HOME/bge-m3-server/results"
cd "$HOME/bge-m3-server"
lscpu > results/lscpu.txt
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE > results/cpu-topology.txt
numactl --hardware > results/numa.txt
free -h > results/memory.txt
uname -a > results/kernel.txt
python -m pip freeze > results/packages.txt
```

鲲鹏官方材料确认 950 支持 SVE2，但**SVE2 不能代替 I8MM 或 SME2 的检查**。查看 `lscpu` 的 Features/Flags 和 `/proc/cpuinfo`，并让 `PreparedW8A8I8MM` 的硬件检查通过。如果没有 I8MM，当前专用 W8 路径不能直接运行：先测 BF16 原生基线，后续增加匹配指令的算子；不要强制打开 I8MM 标志绕过检查。现有 SME 目标只按已验证 Apple 配置构建，不把 `-mcpu=apple-m4` 用到鲲鹏。

从 `cpu-topology.txt` 选择一个 NUMA 节点及其物理核列表，每物理核先取一个逻辑 CPU，并确保在进程/容器允许的 cpuset 内。**不假设每节点 CPU 编号连续，也不把 192/4=48 直接写成拓扑事实。** 后文 `$CPUS` 与 `$NODE` 用实际结果填写。

CPU 环境基线为匹配的 Linux aarch64 Python 3.11、PyTorch 2.11 CPU、vLLM 0.24 CPU、Triton CPU 3.7.2 系列与上述 CPU Raw 能力。先准备该环境，再装本分支；不提供一个未经验证的社区 `pip install` 命令冒充整套优化已安装。Torch 和扩展必须 ABI 匹配，OpenMP 必须链接 Torch 的同一 runtime。

## 5 获取代码和构建

以下命令从已激活的匹配 CPU Python 环境开始，要求 Git、CMake≥3.25、Ninja、C++ 编译器与 Torch CMake 配置已存在。不要使用 Mac 安装包或直接运行锁死 P1 八个核的 `run-cix.sh`。

```bash
export WORK_ROOT="$HOME/bge-m3-server"
cd "$WORK_ROOT"
git clone -b bge-m3 https://github.com/kevinzs2048/FlagGems.git
git clone -b bge-m3 https://github.com/kevinzs2048/vllm-plugin-FL.git
git clone -b bge-m3 https://github.com/kevinzs2048/flagtree-cpu.git
for repo in FlagGems vllm-plugin-FL flagtree-cpu; do
  git -C "$repo" rev-parse HEAD > "results/$repo.commit"
done
```

如环境尚无对应 CPU Raw 编译器，按 `flagtree-cpu` 仓库的 CPU 构建说明安装该分支，不能用 NVIDIA 默认 Triton 包替代。已有匹配预编译编译器时，先复用它，并同时记录编译器包身份；源码 commit 和二进制身份分别保存。

安装 Python 源码和插件注册：

```bash
cd "$WORK_ROOT"
python -m pip install --no-deps --no-build-isolation -e ./FlagGems
python -m pip install --no-deps --no-build-isolation -e ./vllm-plugin-FL
export ARM_SRC="$WORK_ROOT/FlagGems/src/flag_gems/runtime/backend/_arm/tle_raw/csrc"
cmake -S "$ARM_SRC" -B "$WORK_ROOT/build/arm" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v python)" \
  -DFLAGGEMS_BUILD_ARM_W8A8_I8MM=ON \
  -DFLAGGEMS_BUILD_ARM_ELEMENTWISE=ON
cmake --build "$WORK_ROOT/build/arm" \
  --target flag_gems_arm_w8g32_i8mm flag_gems_arm_elementwise -j 8
```

`w8g32` 是历史库文件名，**不改变本模型每输出通道 W8 的量化规则**。上面两个目标是 FlagGems 自有算子；无需为它们另下 KleidiAI。其他原生依赖是否使用 KleidiAI 应按实际动态链接和调用分别记录。

短输入图是第二阶段选项。先不用 `--short-compile` 跑通，再构建对应桥接：

```bash
cmake -S "$WORK_ROOT/vllm-plugin-FL/csrc/cpu" \
  -B "$WORK_ROOT/build/bridge" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v python)" \
  -DFLAGGEMS_ARM_RAW_INCLUDE="$ARM_SRC" \
  -DFLAGGEMS_ARM_RAW_LIBRARY="$WORK_ROOT/build/arm/libflag_gems_arm_w8g32_i8mm.so"
cmake --build "$WORK_ROOT/build/bridge" -j 8
```

桥接当前按 Neon、head_dim=64、scale=1/8 的 vLLM CPU 接口验证；鲲鹏原生 attention 若选了其他 ISA，先保留原生路径，不要删 ISA 检查来强行捕获图。构建失败优先检查实际 Torch ABI、CMake 配置、编译器和 OpenMP 路径，不要覆盖系统库。

## 6 单 NUMA 功能和线程扫描

模型与匹配校准文件提前放好。`$SCALES` 是外置数据，不会随 Git clone 得到：

```bash
export MODEL_DIR="$WORK_ROOT/models/bge-m3"
export SCALES="$WORK_ROOT/models/bge-m3-sq-fit-scales.pt"
export BENCH="$WORK_ROOT/vllm-plugin-FL/examples/bge_m3/benchmark_cpu.py"
test -f "$MODEL_DIR/pytorch_model.bin"
test -f "$SCALES"
sha256sum "$MODEL_DIR/pytorch_model.bin" "$SCALES" > "$WORK_ROOT/results/weights.sha256"
python "$BENCH" --model "$MODEL_DIR" \
  --prepare-inputs "$WORK_ROOT/inputs.jsonl" \
  --lengths 32 64 128 256 512 1024 2048 4096 8192
```

输入 fixture 只生成一次，两个实现用同一文件。它是固定合成文本的 token 输入，适合找线程和矩阵形状的拐点，不冒充真实语料检索评测。完整文本到向量耗时要另加分词/API 计时。

先设 `$NODE` 和 `$CPUS`，例如从实际拓扑选 node0 的物理核，**不要原样填写猜测的 CPU 编号**。下面 `$T` 不能大于选中物理核数或 64。

```bash
export T=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=1
export TOKENIZERS_PARALLELISM=false
numactl --physcpubind="$CPUS" --membind="$NODE" \
  python "$BENCH" --model "$MODEL_DIR" --backend native-bf16 \
  --inputs "$WORK_ROOT/inputs.jsonl" --lengths 32 64 128 256 512 \
  --threads "$T" --warmups 10 --iterations 50 \
  --output "$WORK_ROOT/results/native-n${NODE}-t${T}-r1"
numactl --physcpubind="$CPUS" --membind="$NODE" \
  python "$BENCH" --model "$MODEL_DIR" --backend flagos-w8 \
  --inputs "$WORK_ROOT/inputs.jsonl" --lengths 32 64 128 256 512 \
  --threads "$T" --scales "$SCALES" \
  --raw-library "$WORK_ROOT/build/arm/libflag_gems_arm_w8g32_i8mm.so" \
  --norm-library "$WORK_ROOT/build/arm/libflag_gems_arm_elementwise.so" \
  --warmups 10 --iterations 50 \
  --output "$WORK_ROOT/results/flagos-n${NODE}-t${T}-r1"
```

`NODE`、`CPUS` 未填写时不能开始。每种配置新进程运行，输出目录必须是新目录，防止覆盖。长输入另跑 `--lengths 1024 2048 4096 8192`；先少量确认内存，再增加迭代。每个长度完成即保存原始时延样本、向量和 JSON，中断不覆盖先前完成的长度。

脚本默认 `--cpu-memory-utilization 0.4`，映射到 vLLM 0.24 历史命名的 `gpu_memory_utilization` 参数，在 CPU 后端控制启动内存预算，并未启用 GPU。共享机器若可用内存不足会在启动时报错；按每 NUMA 的可用内存选择该值，并为两组记录同一值。预算值不是实际常驻量，仍应测 RSS/PSS。BF16 向量舍入后 L2 范数不一定严格等于1，脚本记录范数且不额外重归一化；这不能替代检索质量评测。

建议按顺序只扫描必要变量：

1. 单 NUMA 的 T=8/16/24/32/48，只运行硬件实际允许的值。32/64 token 和长输入分开选线程。
2. 固定已选线程，比较 batch=1/2/4/8/16，先用512 token。`batch × length` 必须小于等于 `--max-batched-tokens`；默认8192。不要一开始就跑8192×16。
3. GNU libgomp 再试 `GOMP_SPINCOUNT=0/300000/1000000/10000000`，一轮只改等待策略。LLVM libomp 改测 `KMP_BLOCKTIME`，不能认为 GOMP 参数对它有效。
4. 短输入基础路径稳定后，加入 `--short-compile --attention-library "$WORK_ROOT/build/bridge/libvllm_fl_bge_cpu.so"`，单独核对收益和完整向量。

同 NUMA 内先量线程拐点；不要把更多线程、更多 batch、编译图和更长忙等一次性全打开，导致不能解释收益。

## 7 四 NUMA 与比较基线

四 NUMA 的首选测试方案是 **每 NUMA 一个独立模型实例，各自绑定本地核和内存，分发不同文档**。这是独立实例的数据并行方案，不等于已经验证了 vLLM 内置 DP。每个实例在启动前执行 `numactl --physcpubind=... --membind=...`，再加载模型，保证权重与打包缓冲尽量本地首次分配。

不能把当前插件直接改成 `TP=4`：`CPUPhaseWorker` 明确限制 TP=1、PP=1。BGE 模型较小，四份权重是可测试的选择，但分词器、框架与工作缓冲也会复制；必须记录四实例总 PSS、各节点内存和总 CPU 时间。不能只报告吞吐而忽略额外内存。

整机吞吐按共同测量窗口内完成总文档数/墙钟时长计算。不把四个不同时段的单实例峰值相加冒充整机成绩。四实例数据集按 ID 分片；不要让四个进程重复编码同一份总数据后仍声称处理了四倍独立文档。

| 对照 | 目的 | 精度说明 |
| --- | --- | --- |
| 原生 vLLM BF16 | 验证模型与框架开箱算术、线程和 NUMA 行为 | 保留原生数学路径；不是 W8 的同精度对照 |
| FlagOS W8A8 | 测试量化与算子组合收益 | 每通道 W8、逐 token A8；独立评估检索质量 |
| llama.cpp Q8_0 CPU | 对焦成熟 CPU 实现 | GGUF 块量化，embedding 精度也不同，不能叫量化完全相同 |
| 可选 llama BF16 | 同存储精度补充对照 | 还需匹配 CLS、归一化、输入、线程、上下文和测量边界 |

llama.cpp 需要使用固定 main commit、记录 CMake flags、开启实际硬件支持的指令，确认是否实际调用 KleidiAI。先分别找各自最优配置，再补同线程/同 batch 控制组。若使用其 HTTP embedding 接口，FlagOS 也必须包含 HTTP/分词计时，不能与上述预分词离线 API 时间混比。

正式记录：输入长度/分布、batch、并发、实例数、线程数、CPU 列表、NUMA 内存绑定、P50/P90/P99延迟、篇/秒、input tokens/秒、初始化耗时、峰值和稳态 RSS/PSS。CPU时间/墙钟时间表示核等效消耗，**不是操作系统线程总数**。使用 `pidstat -t`/`perf`/`numastat -p` 核实实际线程、迁移、上下文切换与远端访存。

性能归因按单项开关做对照，不把跨精度和跨 batch 的整模型差距全归给某一个算子。检索质量用同一批 query/corpus，对照 BF16 的 nDCG@10、Recall、Top-k一致性；不同精度不要求向量逐位相同。768篇等历史总样本数不是一次提交的 batch。

## 8 已有收益与资源边界

以下只是 M5/P1 已有应用回放结果，不是鲲鹏 950 成绩，也不是本次新分支的完整性能回归：

| 场景 | FlagOS | llama.cpp | 已有结论 |
| --- | --- | --- | --- |
| M5完整768篇编码，单篇cap512 | batch16约25.4篇/秒 | 最佳已测batch2约16.21篇/秒 | 持续编码吞吐约+57% |
| P1完整768篇编码，单篇cap512 | batch16约8.36篇/秒 | 最佳已测batch1约5.33篇/秒 | 持续编码吞吐约+56.9% |
| P1真实短查询 | 原选定约19.22ms | 修正指令基线约15.96ms | 仍落后，不能宣称所有短输入都领先 |

历史回放包含预先分词输入到 host 向量返回；不含模型加载、分词、HTTP、索引写入或搜索。batch 不同用于比较各自已测较好配置；不等于同 batch 优势。旧 P1 定长表曾存在 llama ISA 基线遗漏，不沿用那张表宣传新服务器收益。

内存审计显示，软件依赖、分词器副本、BF16 embedding、权重布局和分配器保留都会增加 FlagOS/vLLM 占用。移除测试脚本多余分词器后，单512复核 M5约3.68GiB RSS、P1约3.63GiB；不是新服务器的内存预算。M5双布局576MiB、P1主体单布局288MiB；线程局部缓冲会保留最大容量，多实例/多调用线程可能扩大占用。忙等策略消耗额外CPU；不承诺零内存开销、零功耗代价或所有硬件免调参。

## 9 本次交接的验证状态

- 已把选定 Python 源码、I8MM、SME、元素算子与 attention 叶函数收拢到对应仓库，补充显式 opt-in CMake 构建目标；没有把历史实验目录作为运行依赖。
- M5 的新增构建目标及插件 attention 桥接已实际编译；32/512/8192完整向量的重建对照记录在本地交接材料中。
- P1 新 I8MM 构建通过4项现有算子测试，包含独立整数参考、融合norm/FFN、尾部、输出所有权和FakeTensor合同。
- 新服务器基准脚本已有旧平台功能检查；只作为步骤验证，不以少量迭代结果报告性能。没有继续扩大旧平台评测。
- 新分支未完成完整跨平台性能回归、编译器全量重建、完整CI、多模型重载/并发验收；鲲鹏950尚未执行。最终提交hash、原始检查日志和停留位置随交接文档记录。

## 10 参考资料

- [vLLM CPU 绑核与 NUMA 文档](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)：一般部署原则；具体参数以本次0.24源码和实际日志为准。
- [GNU GOMP_SPINCOUNT](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html)：自旋次数的准确含义。
- [鲲鹏950 SVE2 支持资料](https://www.hikunpeng.com/document/detail/zh/kunpengaccel/kserl/simdjson/docs/zh/release_notes.md)：仅支持SVE2事实，不据此推断I8MM/SME。
