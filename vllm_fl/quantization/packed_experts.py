"""Bridge standard packed expert checkpoints to vLLM's CPU INT4 loader."""
import torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

def unpack_expert_weights(weights):
    for name, tensor in weights:
        if '.mlp.experts.' in name:
            if name.endswith('.weight_shape'):
                continue
            if name.endswith('.weight_packed'):
                assert tensor.dtype == torch.int32 and tensor.ndim == 2
                tensor = unpack_from_int32(tensor, 4, torch.Size((tensor.shape[0],tensor.shape[1]*8)))
                name = name.removesuffix('.weight_packed')+'.weight'
        yield name, tensor
