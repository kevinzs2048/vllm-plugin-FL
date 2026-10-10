"""Load checkpoint INT4 experts one at a time into Torch's KAI layout.

The CPU loader normally retains all unpacked INT8 weights until its final
postprocessing pass. This opt-in path produces the identical packed tensors
while bounding temporary storage to checkpoint experts awaiting their scales.
"""
import re
import os
from pathlib import Path

import torch

_EXPERT = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight_packed|weight_scale|weight_shape)$')


class StreamedExperts:
    def __init__(self, model):
        from vllm.model_executor.utils import replace_parameter
        self.layers = {}
        self.pending = {}
        self.seen = set()
        self.completed = set()
        self.pending_bytes = 0
        self.peak_pending_bytes = 0
        self.packed_experts = 0
        self.loaded_names = set()
        mmap_dir = os.environ.get('MINICPM_EXPERT_MMAP_DIR')
        self.mmap_dir = Path(mmap_dir) if mmap_dir else None
        self.mmap_files = {}
        self.cached_bytes = 0
        self.cache_hit = False
        self.cached_layers = set()
        mmap_limit = int(os.environ.get('MINICPM_EXPERT_MMAP_LAYERS', '0'))
        if mmap_limit < 0:
            raise ValueError('MINICPM_EXPERT_MMAP_LAYERS must be nonnegative')
        self.checkpoint = getattr(model.config, '_name_or_path', None) if hasattr(model, 'config') else None
        if self.mmap_dir:
            self.mmap_dir.mkdir(parents=True, exist_ok=True)
        for idx, block in enumerate(model.language_model.model.layers):
            routed = getattr(getattr(block.mlp, 'experts', None), 'routed_experts', None)
            if routed is None:
                continue
            method = routed.quant_method
            if type(method).__name__ != 'CompressedTensorsW4A8Int8MoEMethod' or method.group_size != 32 or method.has_bias:
                raise ValueError('Streamed experts require CPU W4A8 G32 without expert bias')
            e, two_i, h = routed.w13_weight.shape
            if routed.w2_weight.shape != (e, h, two_i // 2):
                raise ValueError('Unexpected expert dimensions')
            self.layers[idx] = (routed, e, h, two_i // 2)
            for name in ('w13_weight', 'w2_weight', 'w13_weight_scale', 'w2_weight_scale'):
                replace_parameter(routed, name, torch.nn.Parameter(torch.empty(0), requires_grad=False))
            original = method.process_weights_after_loading
            def postprocess(layer, _original=original):
                if not getattr(layer, '_fl_streamed_experts_complete', False):
                    raise RuntimeError('Incomplete streamed expert loading')
                # Kernel and packed parameters are already initialized below.
            method.process_weights_after_loading = postprocess
        self.mapped_layers = set(sorted(self.layers)[:mmap_limit] if mmap_limit else self.layers) if self.mmap_dir else set()
        # tcmalloc can retain all the original scale allocations after the
        # placeholders are released. Return those pages before loading data.
        self._release_unused_pages()
        if self.mmap_dir and os.environ.get('MINICPM_EXPERT_CACHE_REUSE') == '1':
            from vllm_fl.quantization.expert_cache import read_manifest
            if not self.checkpoint:
                raise ValueError('Cache reuse requires a local checkpoint identity')
            layouts=[(idx,e,h,inter) for idx,(_,e,h,inter) in self.layers.items() if idx in self.mapped_layers]
            entries=read_manifest(self.mmap_dir,self.checkpoint,layouts)
            if entries is not None:
                for (idx,name),entry in entries.items():
                    path=self.mmap_dir/f'layer-{idx}-{name}.bin'
                    dtype=getattr(torch,entry.get('dtype','torch.uint8').removeprefix('torch.'))
                    size=entry['shape'][0]*entry['shape'][1]
                    data=torch.from_file(str(path),shared=False,size=size,dtype=dtype).reshape(entry['shape'])
                    replace_parameter(self.layers[idx][0],name,torch.nn.Parameter(data,requires_grad=False))
                    self.cached_bytes+=entry['bytes']
                self.cached_layers = set(self.mapped_layers)
                for idx in self.cached_layers:
                    self._activate_layer(idx)
                self.cache_hit=True
                print('[vllm_fl] Reusing verified-layout KAI expert mmap cache',flush=True)

    def _activate_layer(self, idx):
        from vllm.model_executor.layers.fused_moe.oracle.w4a8_int8 import make_w4a8_int8_moe_kernel
        layer=self.layers[idx][0]
        method=layer.quant_method
        method.moe_kernel=make_w4a8_int8_moe_kernel(
            moe_quant_config=method.get_fused_moe_quant_config(layer),
            moe_config=method.moe,experts_cls=method.experts_cls,
            routing_tables=layer._expert_routing_tables())
        layer._fl_streamed_experts_complete=True
        prefix=f'language_model.model.layers.{idx}.mlp.experts.routed_experts.'
        self.loaded_names.update(prefix+n for n,_ in layer.named_parameters())

    @staticmethod
    def _release_unused_pages():
        import ctypes
        library = ctypes.CDLL(None)
        release = getattr(library, 'MallocExtension_ReleaseFreeMemory', None)
        if release is not None:
            release.argtypes = []
            release.restype = None
            release()
        elif os.name == 'posix':
            trim = getattr(library, 'malloc_trim', None)
            if trim is not None:
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                trim(0)

    def _pack_expert(self, key):
        from vllm.model_executor.layers.fused_moe.oracle.w4a8_int8 import pack_int4_weights_for_kleidi
        from vllm.model_executor.utils import replace_parameter
        idx, expert = key
        layer, e, h, inter = self.layers[idx]
        tensors = self.pending.pop(key)
        w13 = torch.cat((tensors['gate_proj', 'weight_packed'], tensors['up_proj', 'weight_packed']), 0)
        s13 = torch.cat((tensors['gate_proj', 'weight_scale'], tensors['up_proj', 'weight_scale']), 0).to(torch.bfloat16)
        w2 = tensors['down_proj', 'weight_packed']
        s2 = tensors['down_proj', 'weight_scale'].to(torch.bfloat16)
        p13 = pack_int4_weights_for_kleidi(w13, s13, None, h, 2 * inter, 32)
        p2 = pack_int4_weights_for_kleidi(w2, s2, None, inter, h, 32)
        if layer.w13_weight_packed.numel() == 0:
            for name, packed in [('w13_weight_packed', p13), ('w2_weight_packed', p2)]:
                if idx in self.mapped_layers:
                    path = self.mmap_dir / f'layer-{idx}-{name}.bin'
                    if path.exists():
                        raise FileExistsError(f'Use a new expert mmap directory: {path}')
                    data = torch.from_file(str(path), shared=True, size=e*packed.numel(), dtype=packed.dtype).reshape(e,-1)
                    self.mmap_files.setdefault(idx, []).append(str(path))
                else:
                    data = torch.empty((e, packed.numel()), dtype=packed.dtype)
                replace_parameter(layer, name, torch.nn.Parameter(data, requires_grad=False))
        layer.w13_weight_packed[expert].copy_(p13.reshape(-1))
        layer.w2_weight_packed[expert].copy_(p2.reshape(-1))
        self.pending_bytes -= sum(t.numel() * t.element_size() for t in tensors.values())
        self.completed.add(key)
        self.packed_experts += 1
        if sum((idx, i) in self.completed for i in range(e)) == e:
            self._activate_layer(idx)
            for path in self.mmap_files.get(idx, ()):
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            self._release_unused_pages()
            print(f'[vllm_fl] Streamed KAI experts: layer {idx} complete ({self.packed_experts} experts)', flush=True)

    def weights(self, weights):
        from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
        for name, tensor in weights:
            match = _EXPERT.fullmatch(name)
            if match is None:
                if '.mlp.experts.' in name:
                    raise ValueError(f'Unsupported streamed expert tensor: {name}')
                yield name, tensor
                continue
            idx, expert = int(match[1]), int(match[2])
            proj, kind = match[3], match[4]
            layer, e, h, inter = self.layers[idx]
            if not 0 <= expert < e:
                raise ValueError(f'Invalid expert index: {name}')
            if kind == 'weight_shape':
                continue
            if name in self.seen or (idx, expert) in self.completed:
                raise ValueError(f'Duplicate expert tensor: {name}')
            self.seen.add(name)
            shape = (h, inter) if proj == 'down_proj' else (inter, h)
            if kind == 'weight_packed':
                if tensor.dtype != torch.int32 or tensor.shape != (shape[0], shape[1] // 8):
                    raise ValueError(f'Invalid packed weight: {name}, {tensor.shape}, {tensor.dtype}')
                if idx not in self.cached_layers:
                    tensor = unpack_from_int32(tensor, 4, torch.Size(shape))
            elif tensor.shape != (shape[0], shape[1] // 32):
                raise ValueError(f'Invalid group scales: {name}, {tensor.shape}')
            if idx in self.cached_layers:
                bucket=self.pending.setdefault((idx,expert),set())
                bucket.add((proj,kind))
                if len(bucket)==6:
                    self.pending.pop((idx,expert));self.completed.add((idx,expert));self.packed_experts+=1
                continue
            # Clone scales so a pending small tensor cannot retain an mmap shard.
            tensor = tensor.contiguous().clone() if kind == 'weight_scale' else tensor
            bucket = self.pending.setdefault((idx, expert), {})
            bucket[proj, kind] = tensor
            self.pending_bytes += tensor.numel() * tensor.element_size()
            self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)
            if self.pending_bytes > 2 * 1024**3:
                raise RuntimeError('Checkpoint order exceeds the 2 GiB streamed loading workspace')
            if len(bucket) == 6:
                self._pack_expert((idx, expert))
        expected = sum(e for _, e, _, _ in self.layers.values())
        if self.pending or self.packed_experts != expected:
            raise ValueError(f'Incomplete experts: {self.packed_experts}/{expected}')
        self.report = dict(experts=self.packed_experts, peak_pending_bytes=self.peak_pending_bytes,
                           mmap_dir=str(self.mmap_dir) if self.mmap_dir else None,
                           cache_hit=self.cache_hit,
                           mapped_layers=sorted(self.mapped_layers), cached_layers=sorted(self.cached_layers),
                           mmap_bytes=self.cached_bytes+sum(Path(p).stat().st_size for paths in self.mmap_files.values() for p in paths))
        if self.mmap_dir and not self.cache_hit and self.checkpoint:
            from vllm_fl.quantization.expert_cache import write_manifest
            write_manifest(self.mmap_dir,self.checkpoint,
                           [(idx,e,h,inter) for idx,(_,e,h,inter) in self.layers.items() if idx in self.mapped_layers],
                           {(idx,name):getattr(layer,name).dtype for idx,(layer,_,_,_) in self.layers.items()
                            if idx in self.mapped_layers for name in ('w13_weight_packed','w2_weight_packed')})
