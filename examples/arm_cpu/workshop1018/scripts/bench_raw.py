"""Journalled CIX workshop comparison; stock, model-only, and FlagOS routes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

# The release Docker base retains empty experiment variables. Remove empty
# numeric knobs while preserving an explicitly empty plugin allow-list.
for key, value in list(os.environ.items()):
    if not value and key != 'VLLM_PLUGINS' and key.startswith(('VLLM_', 'FL_', 'FLAGGEMS_', 'MINICPM_', 'OMP_')):
        os.environ.pop(key)

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['stock', 'minimal', 'flagos'], required=True)
p.add_argument('--model', default='/model')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--repeats', type=int, default=5)
p.add_argument('--tokens', type=int, default=128)
p.add_argument('--profile', action='store_true')
p.add_argument('--route-audit', action='store_true')
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
a.output.parent.mkdir(parents=True, exist_ok=True)
report = {'complete': False, 'passed': False, 'mode': a.mode, 'model': a.model,
          'variant': os.environ.get('BUILD_VARIANT', 'release' if a.mode == 'flagos' else 'native-prefix'),
          'runs': [], 'cases': [], 'protocol': 'batch=1, fixed512 input, greedy, prefix cache off, warmup excluded',
          'environment': {k: v for k, v in os.environ.items() if k.startswith(('FL_', 'FLAGGEMS_', 'VLLM_', 'OMP_', 'MINICPM_', 'MIMALLOC_', 'TCMALLOC_'))},
          'checkpoint': {n: hashlib.sha256((Path(a.model)/n).read_bytes()).hexdigest()
                         for n in ['config.json', 'model.safetensors.index.json']}}
def save():
    temp = a.output.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    temp.replace(a.output)
save()
try:
    import torch
    from collections import Counter
    from flag_gems.runtime.backend._arm.fused import native_ops as _arm_native_api
    _raw_python_calls = Counter()
    _audit_active = False
    _raw_getattr = _arm_native_api._RawOps.__getattr__
    def _audited_raw_getattr(self, name):
        target = _raw_getattr(self, name)
        def invoke(*args, **kwargs):
            if _audit_active:
                _raw_python_calls[str(target)] += 1
            return target(*args, **kwargs)
        return invoke
    _arm_native_api._RawOps.__getattr__ = _audited_raw_getattr
    from flag_gems.runtime.backend._arm.gdn import conv_prefill as _conv_api
    _triton_python_calls = Counter()
    def _watch_triton_conv(name):
        target = getattr(_conv_api, name)
        def invoke(*args, **kwargs):
            if _audit_active:
                _triton_python_calls[name] += 1
            return target(*args, **kwargs)
        setattr(_conv_api, name, invoke)
    for _name in ('gdn_conv1d_prefill_triton', 'gdn_conv1d_update_triton'):
        _watch_triton_conv(_name)
    from importlib.metadata import version
    from vllm import LLM, ModelRegistry, SamplingParams
    report['versions'] = {n: version(n) for n in ['vllm', 'torch', 'transformers']}
    report['stock_architecture_registered'] = 'MiniCPMV4_7ForConditionalGeneration' in ModelRegistry.get_supported_archs()
    if a.mode == 'minimal':
        # Avoid the product plugin's __init__/general-entrypoint side effects.
        # Use its model definitions only, without loading FlagGems arithmetic.
        import types, importlib.machinery
        package = types.ModuleType('vllm_fl')
        package.__path__ = [os.environ.get('MODEL_PLUGIN_SOURCE', '/opt/flagos-runtime/venv/lib/python3.11/site-packages/vllm_fl')]
        package.__spec__ = importlib.machinery.ModuleSpec('vllm_fl', loader=None, is_package=True)
        sys.modules['vllm_fl'] = package
        # Model/processor registration and CPU ABI correctness only.
        # Keep stock vLLM int8, GDN, norm, attention, worker and output head.
        from vllm_fl.models.minicpmv4_7.registration import register_model
        register_model()
        from vllm_fl.models.minicpmv4_7.model import MiniCPMV4_7ForConditionalGeneration, MiniCPMV4_7DenseForConditionalGeneration
        ModelRegistry.register_model('MiniCPMV4_7ForConditionalGeneration', MiniCPMV4_7ForConditionalGeneration)
        ModelRegistry.register_model('MiniCPMV4_7DenseForConditionalGeneration', MiniCPMV4_7DenseForConditionalGeneration)
        from vllm_fl.patches.arm_cpu_gdn import apply_arm_cpu_gdn_state_indices_patch
        apply_arm_cpu_gdn_state_indices_patch()
        from vllm_fl.patches.arm_minicpm_runtime import _install_cpu_gemm_guard, _install_cpu_cleanup_guard, _install_cpu_hybrid_kv_zeroing, _install_cpu_sampling_rng
        assert os.environ.get('FL_INT8_LMHEAD', '0') == '0'
        _install_cpu_gemm_guard()
        _install_cpu_cleanup_guard()
        # CPU metadata/lifecycle compatibility, independent of Triton/FlagGems.
        # Use the stock zeroer's segment addresses and logical page size.
        import ctypes
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner
        def cpu_zero_pages(self, block_ids):
            zeroer = getattr(self, '_kv_block_zeroer', None)
            meta = getattr(zeroer, '_meta', None)
            if not block_ids or meta is None:
                return
            assert self.device.type == 'cpu'
            addresses, page_size_el, _, _ = meta
            page_bytes = int(page_size_el)*4
            for address in addresses.tolist():
                for block in block_ids:
                    assert block >= 0
                    ctypes.memset(int(address)+int(block)*page_bytes, 0, page_bytes)
        CPUModelRunner._zero_block_ids = cpu_zero_pages
        from vllm.v1.worker import block_table
        from vllm.utils.cpu_triton_utils import compute_slot_mapping_kernel
        block_table._compute_slot_mapping_kernel = compute_slot_mapping_kernel
        _install_cpu_sampling_rng()
        report['minimal_hooks'] = ['model/processor registration', 'non-2D CPU GEMM lifecycle guard', 'GDN contiguous state indices', 'CPU cache cleanup/zeroing', 'native CPU slot mapping', 'CPU sampling RNG']
    from vllm.model_executor.kernels.linear.scaled_mm.cpu import CPUInt8ScaledMMLinearKernel
    report['w8_apply_module'] = CPUInt8ScaledMMLinearKernel.apply_weights.__module__
    save()
    start = time.perf_counter()
    opts = dict(model=a.model, trust_remote_code=True, dtype='bfloat16',
                max_model_len=4096, max_num_batched_tokens=512, max_num_seqs=1,
                kv_cache_memory_bytes=268435456, mamba_ssm_cache_dtype='float32',
                distributed_executor_backend='uni', enforce_eager=True,
                enable_prefix_caching=False, generation_config='vllm',
                disable_log_stats=False,
                limit_mm_per_prompt={'image': 0, 'video': 0})
    if a.mode == 'flagos':
        opts['worker_cls'] = 'vllm_fl.worker.cpu_phase_worker.CPUPhaseWorker'
    llm = LLM(**opts)
    report['load_seconds'] = time.perf_counter()-start
    report['flag_gems_modules_loaded'] = sum(n.startswith('flag_gems') for n in sys.modules)
    if a.mode == 'minimal':
        assert report['flag_gems_modules_loaded'] == 0
        assert CPUInt8ScaledMMLinearKernel.apply_weights.__module__.startswith('vllm.model_executor.')
    report['w8_apply_module_after_load'] = CPUInt8ScaledMMLinearKernel.apply_weights.__module__
    tokenizer = llm.get_tokenizer()
    ids = tokenizer.encode('请解释机器学习的训练过程，并举出具体例子。'*100, add_special_tokens=False)[:512]
    assert len(ids) == 512
    for i in range(a.repeats+1):
        out = llm.generate([{'prompt_token_ids': ids}], SamplingParams(temperature=0, max_tokens=a.tokens, min_tokens=a.tokens, ignore_eos=True), use_tqdm=False)[0]
        m = out.metrics
        assert not m.is_corrupted and len(out.outputs[0].token_ids) == a.tokens
        t0, t1, t2 = m.scheduled_ts, m.first_token_ts, m.last_token_ts
        row = dict(repeat=i, warmup=i==0, pp512=512/(t1-t0),
                   tg128=(a.tokens-1)/(t2-t1), total=(512+a.tokens)/(t2-t0),
                   prefill_s=t1-t0, decode_s=t2-t1, execution_s=t2-t0,
                   token_ids=list(out.outputs[0].token_ids))
        report['runs'].append(row); save()
        print('RUN', json.dumps({k:v for k,v in row.items() if k!='token_ids'}), flush=True)
    for question, expected in [('只输出结果：2+3等于多少？', '5'), ('中国的首都是哪里？请只回答城市名。', '北京')]:
        prompt = tokenizer.apply_chat_template([{'role':'user','content':question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out = llm.generate([prompt], SamplingParams(temperature=0,max_tokens=32),use_tqdm=False)[0]
        text = out.outputs[0].text
        report['cases'].append(dict(question=question, text=text, passed=expected in text)); save()
        assert expected in text
    if a.route_audit:
        from collections import Counter
        from torch.utils._python_dispatch import TorchDispatchMode
        from vllm_fl.ops.arm_operator_backend import SELECTED
        from vllm_fl.ops.minicpmv47_gdn_prefill import CALLS as _gdn_prefill_calls
        from flag_gems.runtime.backend._arm.gdn.inference import reset_route_stats, route_stats
        seen = Counter()
        class AuditMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                seen[str(func)] += 1
                return func(*args, **(kwargs or {}))
        reset_route_stats()
        _gdn_prefill_calls.clear()
        _audit_active = True
        with AuditMode():
            audited = llm.generate([{'prompt_token_ids':ids}], SamplingParams(temperature=0,max_tokens=8,min_tokens=8,ignore_eos=True),use_tqdm=False)[0]
        _audit_active = False
        arm = {k:v for k,v in seen.items() if k.startswith('flag_gems_arm_')}
        allowed_host = ('set_', 'get_', 'register_')
        illegal = {k:v for k,v in arm.items() if not k.endswith(('.raw','.triton')) and not k.split('.')[1].startswith(allowed_host)}
        report['route_audit'] = dict(selected_backends=dict(SELECTED), arm_inference_calls=arm,
             other_operator_calls=dict(seen), gdn_routes=route_stats(), rejected_direct_arm_calls=illegal, raw_python_calls=dict(_raw_python_calls),
             triton_python_calls=dict(_triton_python_calls),
             plugin_gdn_prefill_calls=dict(_gdn_prefill_calls),
             audit_output_tokens=len(audited.outputs[0].token_ids), scope='FlagOS ARM inference operators; framework ops listed separately')
        save()
        assert arm and not illegal, illegal
        assert any('gdn_prefill.raw' in k for k in _raw_python_calls), _raw_python_calls
        assert _triton_python_calls['gdn_conv1d_prefill_triton'] > 0, _triton_python_calls
        assert not any('fallback' in k and v for k,v in _gdn_prefill_calls.items()), _gdn_prefill_calls
        assert not any(k.endswith('.default') and not k.split('.')[1].startswith(allowed_host) for k in _raw_python_calls), _raw_python_calls
        assert all(v=='raw' for v in SELECTED.values()), SELECTED
        assert any('w8a8.linear.raw' in k for k in arm), arm
        assert any('gdn_' in k and k.endswith('.raw') for k in arm), arm
        assert not any(k.startswith('fallback_') and v for k,v in route_stats().items()), route_stats()
    if a.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
            llm.generate([{'prompt_token_ids':ids}], SamplingParams(temperature=0,max_tokens=16,min_tokens=16,ignore_eos=True),use_tqdm=False)
        prof.export_chrome_trace(str(a.output.with_suffix('.trace.json')))
        report['profile_top'] = [{'name':e.key,'self_cpu_us':e.self_cpu_time_total,'cpu_us':e.cpu_time_total,'calls':e.count} for e in sorted(prof.key_averages(),key=lambda e:e.self_cpu_time_total,reverse=True)[:25]]
    rows = report['runs'][1:]
    report['medians'] = {k:statistics.median(r[k] for r in rows) for k in ['pp512','tg128','total']} if rows else None
    report.update(complete=True, passed=True); save()
except BaseException as error:
    report['error'] = repr(error); save(); raise
