CIX P1 / MiniCPM-V-4.7-1B W8A8 / FlagOS：新板卡在线部署指南
2026-10-17；所有大文件由新环境自行下载，不需要连接M5，也不需要7GB完整离线包。

一、究竟需要下载什么

必需：
1. Docker基础镜像：Harbor；含Python3.11.2、PyTorch2.11.0+cpu、vLLM0.24.0+cpu、
   Transformers5.15.1、libtriton_jit、OpenMP和运行时编译/链接工具。
2. 模型：ModelScope的FlagRelease/MiniCPM-V-4.7-1B-W8A8-arm-FlagOS，约2.63GB。
   下载完整仓库，保留权重分片、配置、Tokenizer、Processor和模型Python文件。
3. 预编译包：FlagOS-CIX-P1-prebuilt-cp311-torch211-20261016-r1.tar.gz，174,323,622字节。
   包含FlagTree-CPU、FlagGems与vLLM-plugin-FL三个wheel，以及安装检查工具。
4. 轻量Workshop包：FlagOS-CIX-P1-online-workshop-20261017-r2.tar.gz，约0.4MB。
   包含启动/检查/HTTP/性能脚本、固定版本与模型哈希清单、示例图片、OCR字体及许可证。
   它不含模型、Docker层、LLVM SDK、源码仓库或预编译wheel。

源码：
三份模型适配/算子源码均已推到kevinzs2048个人GitHub仓库的固定提交，并非已经合入社区main：
FlagGems         b5f5e110c7c2ae40521d0de897f3961467265fa5
flagtree-cpu     36133df9540cdd9eb5b09209ee41ce1697b27126
vllm-plugin-FL   f0de702335902facfffe625c0690f7a3df7f617d

只运行FlagOS推理时，源码仓库可以不下载，因为三个组件使用已编译wheel。
做源码讲解/检查可拉齐三份仓库；做“原生最小适配基线”对比至少要下载上述plugin源码。
新轻量Workshop脚本也在vllm-plugin-FL仓库examples/arm_cpu/workshop1018中，
使用独立的workshop1018-online-20261017分支与本指南配套release标签。

默认不需要另下：
- LLVM SDK、nlohmann/json头文件、Sleef源码、Python源码构建依赖：仅源码编译路线需要。
- KleidiAI SDK、源码或静态库：当前FlagGems自有TLE Raw路径不需要。
- PyTorch、vLLM、libtriton_jit、OpenMP安装包：镜像已经包含，不在主机重复pip安装。
- JIT缓存：首次在各板卡生成，不能把“预编译wheel”理解成完全没有首次JIT。

本指南使用同一基础镜像加新wheel来复现Triton/TLE Raw路径。
只下载旧Harbor镜像并直接使用其内置默认FlagGems，不能视作复现了本指南的算子路径。

二、主机条件与固定目录

账号baai，HOME=/home/baai。Linux aarch64，16 GB或以上物理内存（系统可见内存至少15 GiB）。首次启动仍会生成JIT缓存。
八个Cortex-A720：0,1,6,7,8,9,10,11。不同CPU编号的板卡要先重新确认亲和性。
建议新板至少25GiB空闲磁盘；Docker数据目录若在另一块盘，也要留出镜像下载/解压空间。
主机先具备Docker daemon、sudo -n docker权限、Python>=3.9、python3-venv、Git、curl、tar、gzip。
Docker安装请按主机发行版使用官方文档：https://docs.docker.com/engine/install/
不把主机Docker安装、sudo权限配置或账号创建冒充为本软件包自动完成的事项。

后续全部操作在新P1主机执行，不是在M5执行。
每个代码块按顺序单独执行；遇到任何错误先停止，不能忽略错误继续下一步。

三、建立空工作目录并下载轻量脚本

本段使用一个新的主机Bash终端。set -e使失败时停止，避免覆盖或继续使用不完整材料。
如果已有/home/baai/workshop1018，先检查用途并归档本项目旧目录，勿直接覆盖解包。

set -euo pipefail
export WORK_ROOT=/home/baai/workshop1018
test ! -e "$WORK_ROOT"
mkdir "$WORK_ROOT"
mkdir "$WORK_ROOT/downloads"
cd "$WORK_ROOT"
export ONLINE_RELEASE='https://github.com/kevinzs2048/vllm-plugin-FL/releases/download/cix-p1-online-workshop-20261017-r2'
export KIT=FlagOS-CIX-P1-online-workshop-20261017-r2.tar.gz
curl -fL --retry 3 "$ONLINE_RELEASE/$KIT" -o "downloads/$KIT"
curl -fL --retry 3 "$ONLINE_RELEASE/$KIT.sha256" -o "downloads/$KIT.sha256"
(cd downloads && sha256sum -c "$KIT.sha256")
tar -xzf "downloads/$KIT" -C "$WORK_ROOT"
source "$WORK_ROOT/scripts/activate_host.env"

新主机终端需重新执行export WORK_ROOT和source activate_host.env。
脚本从工作目录确定路径，不依赖M5、任何固定开发板IP或/home/kevin路径。

四、从Harbor拉取固定镜像

如仓库需要鉴权，先用获授权账号登录；本指南不保存密码：

sudo docker login harbor.baai.ac.cn

镜像名称、不可变digest和image ID已经写在脚本的release.env中：

source "$WORK_ROOT/scripts/release.env"
sudo docker pull "$CIX_IMAGE_REF"
test "$(sudo docker image inspect "$CIX_IMAGE_REF" --format '{{.Id}}')" = "$CIX_IMAGE_ID"
sudo docker image inspect "$CIX_IMAGE_REF" --format 'id={{.Id}} repo-digests={{json .RepoDigests}}'

固定registry digest：sha256:cb9fc2bb4bdda738d72367ce01e52347a180371da0f4967864c74848ea77a261
固定image ID：sha256:72d655d6bf308074aa53374ae32e7f1596d0493b81c1960df0209994bc54816b
镜像本身不包含本次模型权重。

五、从ModelScope下载模型

下载工具装在主机独立venv里，不改变推理镜像的Python环境：

python3 -m venv "$HOME/.venvs/modelscope-download"
"$HOME/.venvs/modelscope-download/bin/python" -m pip install modelscope
export MODEL_DIR="$WORK_ROOT/models/MiniCPM-V-4.7-1B-W8A8-FlagOS"
"$HOME/.venvs/modelscope-download/bin/modelscope" download \
  --model FlagRelease/MiniCPM-V-4.7-1B-W8A8-arm-FlagOS \
  --revision master \
  --local_dir "$MODEL_DIR"

模型页面：https://modelscope.cn/models/FlagRelease/MiniCPM-V-4.7-1B-W8A8-arm-FlagOS
本指南校验23个运行必需文件的固定SHA256；README等说明文件更新不会误报成权重变化。
若将来master换了权重/配置，检查会拒绝继续，不会悄悄把新权重当成本次已测版本。
不能替换成原始BF16、GGUF、AWQ或另一个MiniCPM5模型。

六、下载约174MB预编译包

cd "$WORK_ROOT"
export WHEEL_RELEASE='https://github.com/kevinzs2048/FlagGems/releases/download/cix-p1-workshop-20261016-r1'
export WHEEL_PKG=FlagOS-CIX-P1-prebuilt-cp311-torch211-20261016-r1.tar.gz
curl -fL --retry 3 "$WHEEL_RELEASE/$WHEEL_PKG" -o "downloads/$WHEEL_PKG"
printf '%s  %s\n' \
  a53ea331abca38cc75e2192f8291ac8630b2c06451bd3e0a40cdc9b51ef229d4 \
  "downloads/$WHEEL_PKG" | sha256sum -c -
test ! -e "$WORK_ROOT/prebuilt"
tar -xzf "downloads/$WHEEL_PKG" -C "$WORK_ROOT"

它会生成prebuilt/wheels和三个安装/检查脚本。三个wheel为：
triton-3.7.2+git36133df9-cp311-cp311-linux_aarch64.whl
flag_gems-0.0+gb5f5e110c-cp311-cp311-linux_aarch64.whl
vllm_plugin_fl-0.0.0+gf0de70233-py3-none-any.whl

FlagTree-CPU的pip包名是triton。不要改成在主机pip install triton，也不要用任意PyPI最新版替代。
FlagGems平台wheel包含自有Raw动态库、SiLU bitcode与许可证。
TLE Raw内部执行FlagGems自有C++/Neon例程，不需要另行获取KleidiAI。

七、下载源码（源码讲解/原生对比时执行）

只做FlagOS HTTP可跳过本段。完整Workshop与性能对比建议执行：

bash "$WORK_ROOT/scripts/fetch_sources.sh"

脚本从以下GitHub仓库拉取并锁定本指南第一节的完整commit，不使用漂移的main：
https://github.com/kevinzs2048/FlagGems
https://github.com/kevinzs2048/flagtree-cpu
https://github.com/kevinzs2048/vllm-plugin-FL

如果只做原生最小适配对比，至少需要：

mkdir -p "$WORK_ROOT/source"
git clone --filter=blob:none --no-checkout \
  https://github.com/kevinzs2048/vllm-plugin-FL.git \
  "$WORK_ROOT/source/vllm-plugin-FL"
git -C "$WORK_ROOT/source/vllm-plugin-FL" fetch origin f0de702335902facfffe625c0690f7a3df7f617d
git -C "$WORK_ROOT/source/vllm-plugin-FL" checkout --detach f0de702335902facfffe625c0690f7a3df7f617d

上面“完整拉取”和“只拉plugin”二选一，不在已有仓库上重复git clone。
源码仓库只用于阅读、可选构建或minimal模型注册，不等于必须现场重新编译。

八、主机检查

source "$WORK_ROOT/scripts/activate_host.env"
bash "$WORK_ROOT/scripts/check_preloaded.sh"

需要看到ONLINE_INPUTS_CHECK_PASS、PREBUILT_CHECK_PASS、PRELOADED_CHECK_PASS。
它核验平台、23个模型文件、脚本/字体、已下载源码提交、镜像身份和wheel哈希。
此在线版本不要求原来完整离线包的PRELOADED-MANIFEST.json、LLVM SDK或wheelhouse。
已有源码如果提交错误或工作区有改动会拒绝继续。

九、两个容器的职责

flagos-workshop1018：开发操作容器。主进程sleep infinity；负责pip安装、算子验收和性能测试。
minicpm-1b-workshop1018：HTTP推理服务容器。主进程vLLM；负责文字/图片API。
两者用同一基础镜像、同一模型和同一/work/envs/dev安装结果，无需安装两次。
开发容器中exit只退出终端；后面docker stop停止的是另一个HTTP服务容器。
性能测试前停服务，释放模型占用的CPU和内存，开发容器继续保留。

十、创建开发容器并安装wheel

下面先在P1主机执行，最后一条会进入开发容器：

CONTAINER=flagos-workshop1018 bash "$WORK_ROOT/scripts/init_container.sh"
sudo docker exec -it flagos-workshop1018 /bin/bash

以下在开发容器执行（路径改为/work和/model，Python统一使用dev_python.sh）：

set -euo pipefail
/opt/flagos-runtime/bin/flagos-python --profile 1b /work/scripts/prepare_env.py
bash /work/scripts/dev_python.sh /work/scripts/prebuilt_tools.py preflight /work
bash /work/scripts/dev_python.sh -m pip install --no-index --no-deps \
  --no-compile --target /work/envs/dev/lib/python3.11/site-packages \
  /work/prebuilt/wheels/*.whl
bash /work/scripts/dev_python.sh /work/scripts/prebuilt_tools.py installed /work

成功标志：PREBUILT_IMPORT_AND_RAW_LOAD_PASS。
此步骤使用本地wheel，不联网，不重新编译安装FlagTree/FlagGems。
不要重复初始化已有前缀；发生中断时保留日志，归档本次失败前缀后重新初始化。

十一、算子和模型验收（开发容器）

bash /work/scripts/validate_source.sh /work/evidence/online-run01

检查CPU JIT、Raw动态库、8组W8数值、真实Triton/TLE Raw调用、算术/中文/图片/OCR。
结果与日志保存在/work/evidence/online-run01，主机对应WORK_ROOT/evidence/online-run01。
全通过输出SOURCE_VALIDATION_PASS。首次需要JIT和权重准备，可能数分钟。
已有目录会拒绝覆盖；重测用online-run02等新名字。

十二、回到主机并启动HTTP服务

在开发容器单独执行这一行，等回到baai主机提示符：

exit

下面只在主机执行：

export WORK_ROOT=/home/baai/workshop1018
source "$WORK_ROOT/scripts/activate_host.env"
export CACHE_DIR="$WORK_ROOT/cache/http-online01"
export RUN_DIR="$WORK_ROOT/evidence/http-online01"
export CONTAINER=minicpm-1b-workshop1018
bash "$WORK_ROOT/scripts/start_service.sh"

等最终READY。冷启动轮询中的连接拒绝可以是服务仍在初始化，以最终READY/超时结果为准。
上下文4096、batch token512、单sequence、单图片、KV256MiB、GDN FP32、DSpark关闭。
服务监听127.0.0.1:8000，在P1主机本地访问。

十三、文字与图片端到端请求（主机）

curl --noproxy '*' --fail http://127.0.0.1:8000/v1/models
python3 "$WORK_ROOT/scripts/http_request.py" --output "$RUN_DIR/text-response.json"
python3 "$WORK_ROOT/scripts/http_request.py" \
  --image "$WORK_ROOT/scripts/example.png" \
  --text '这张图片的主要颜色是什么？' \
  --output "$RUN_DIR/image-response.json"
sudo docker logs "$CONTAINER" > "$RUN_DIR/server-full.log" 2>&1

文字应答5；图片应识别红色。保存request、response、usage和HTTP秒数；不能把HTTP时间当TG128。
已有输出会拒绝覆盖，重试换文件名，不抹掉旧结果。
默认生成temperature=0、enable_thinking=false、max_tokens=128。

十四、原生基线与FlagOS性能对照（可选，须先下载plugin源码）

先在主机停止HTTP服务，再进入仍运行的开发容器。这两行针对的是两个不同容器：

sudo docker stop -t 30 minicpm-1b-workshop1018
sudo docker exec -it flagos-workshop1018 /bin/bash

进入开发容器后执行：

set -euo pipefail
export E=/work/evidence/online-run01
export VLLM_PLUGINS=''
bash /work/scripts/run_bench.sh stock "$E/stock.json" --repeats 0 || \
  bash /work/scripts/dev_python.sh /work/scripts/check_stock.py "$E/stock.json"
bash /work/scripts/run_bench.sh minimal "$E/minimal.json" --repeats 5
bash /work/scripts/run_bench.sh flagos-dev "$E/flagos.json" --repeats 5
bash /work/scripts/dev_python.sh /work/scripts/compare_bench.py \
  "$E/minimal.json" "$E/flagos.json" --output "$E/comparison.json"

stock预期STOCK_EXPECTED_UNSUPPORTED：该vLLM0.24尚未注册模型架构，性能记N/A，不能记0或无限加速。
minimal是可运行的最小适配基线：原生CPU算子、BF16输出头、prefill/decode线程8/8。
FlagOS使用Triton/TLE Raw、W8A8输出头、线程8/4。整体差异包含输出头与线程配置，不冒充单算子收益。
每组1次预热+5次正式、固定512输入/128输出、batch1、贪心、ignoreEOS、prefix cache关闭。
PP512=512/prefill_s，TG128=127/decode_s，Total=640/execution_s，各自取完整正式请求的中位数。
请求逐次保存；partial不作为正式结果，也不拼接不同轮次的最佳PP和TG。

十五、仅在需要“从源码编译组件”时再下载

本指南默认wheel安装路线不用这些材料。若选择编译，请使用独立新环境，不覆盖已装wheel的dev前缀。
编译器SDK：
https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-87717bf9-ubuntu-arm64.tar.gz
nlohmann/json：v3.11.3，https://github.com/nlohmann/json/tree/v3.11.3
FlagTree的Sleef子模块：93f04d869471ce4d007abaebb8c6a7bc62749f61；需git submodule update --init --recursive。
构建依赖包括setuptools76.1.0/77.0.3、setuptools-scm9.2.0、wheel0.46.2、pybind11 3.0.3、cmake3.31.6等。
源码构建还需按仓库构建要求补齐离线/在线依赖，不能把下载了三个仓库说成已经拥有完整编译环境。
本在线轻量包不包含完整源码构建脚本/SDK交付；本节是额外下载边界，不是另一条已打包编译教程。

十六、网络和失败处理

- Harbor鉴权使用组织者提供的授权账号；ModelScope如需授权，按账号权限登录。
- GitHub速度取决于新环境网络，可在当地配置可用的https_proxy/HTTP代理后下载。
  不要求代理指向M5，也不写任何固定开发板IP。无法直连时，可由同城电脑下载小包后转给板卡。
- 大文件中断后，可对同一个目标文件使用curl -C -继续，完成后必须重新校验SHA256。
- 不要在已有目录反复覆盖解包。检查失败先停止，保留日志和partial结果。
- 8000端口或同名容器冲突会拒绝启动；只停止明确属于本项目的服务。
- 服务失败查RUN_DIR/server.log；算子/模型失败查evidence/online-run01里的分项日志。
- 现有项目的代码、环境、模型和测试结果不会由本指南主动删除。

主要下载来源：
ModelScope模型：https://modelscope.cn/models/FlagRelease/MiniCPM-V-4.7-1B-W8A8-arm-FlagOS
预编译包：https://github.com/kevinzs2048/FlagGems/releases/tag/cix-p1-workshop-20261016-r1
在线脚本/本指南：https://github.com/kevinzs2048/vllm-plugin-FL/releases/tag/cix-p1-online-workshop-20261017-r2


本次 r2 材料补充
修正版脚本包接受16 GB板，包含ONLINE lock兼容的17项边界检查；模型、基础镜像和174MB预编译wheel包保持原版本。
性能测试结束并停止HTTP服务后，在P1主机运行：
python3 "$WORK_ROOT/scripts/test_workshop_boundaries.py" --output "$WORK_ROOT/evidence/boundaries-run01.json"
必须显示 BOUNDARIES 17 / 17；已有结果换新文件名。
本次发布不包含 build_source.sh、LLVM SDK 或源码构建依赖，现场只按预编译路线。
