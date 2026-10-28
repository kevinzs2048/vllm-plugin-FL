# Copyright (c) 2026 BAAI. All rights reserved.
"""vLLM checkpoint lifecycle integration for FlagGems ARM KAI operators.

Optional FL_ARM_NATIVE=1: arithmetic and weight layout belong to FlagGems.
Unsupported quantization contracts retain vLLM's original implementation.
"""

import os

import torch


def enabled():
    value = os.environ.get("FL_ARM_NATIVE", "0")
    if value not in ("0", "1"):
        raise ValueError("FL_ARM_NATIVE must be 0 or 1")
    return value == "1"


def prepare_g32(kernel, layer):
    """Return whether this packed checkpoint was consumed by the native path."""
    if not enabled():
        return False
    config = kernel.config
    k, n = config.partition_weight_shape
    if (
        config.group_size != 32
        or config.zero_points
        or config.has_g_idx
        or n < 128
        or n % 64
        or k % 32
    ):
        return False
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

    from vllm_fl.ops.arm_operator_backend import flaggems_ops

    weight = getattr(layer, kernel.w_q_name).detach()
    scales = getattr(layer, kernel.w_s_name).detach()
    if weight.dtype == torch.int32:
        weight = unpack_from_int32(weight, 4, torch.Size((n, k)))
    if (
        weight.dtype != torch.int8
        or weight.shape != (n, k)
        or scales.shape != (n, k // 32)
    ):
        raise ValueError("Unexpected packed G32 checkpoint parameters")
    operators = flaggems_ops("w4a8_g32")
    fraction = float(
        os.environ.get(
            "FL_ARM_SME_FRACTION", os.environ.get("MINICPM_SME_FRACTION", "0")
        )
    )
    if not 0 <= fraction <= 1:
        raise ValueError("FL_ARM_SME_FRACTION must be in [0,1]")
    split = int(n * fraction) // 64 * 64
    # Like vLLM's Dynamic4bitLinearKernel, KAI groupwise execution uses
    # dynamic asymmetric A8 even when the checkpoint declares symmetric A8.
    # MPLinearLayerConfig does not carry an activation-symmetry setting.
    weight = weight.contiguous()
    scales = scales.to(torch.bfloat16).contiguous()
    packed = operators.pack(weight, scales, 0)
    layer._fl_arm_g32_packed = torch.nn.Parameter(packed, requires_grad=False)
    if split:
        layer._fl_arm_g32_prefill = torch.nn.Parameter(
            operators.pack(weight, scales, split), requires_grad=False
        )
    setattr(layer, kernel.w_q_name, None)
    setattr(layer, kernel.w_s_name, None)
    kernel._fl_arm_g32 = (operators, n, k, split)
    return True


def apply_g32(kernel, layer, x, bias):
    operators, n, k, split = kernel._fl_arm_g32
    workers = torch.get_num_threads()
    sme_workers = int(
        os.environ.get(
            "FL_ARM_SME_THREADS", os.environ.get("MINICPM_SME_THREADS", "-1")
        )
    )
    # Keep decode on Neon. SME2/Neon co-run is an explicit prefill policy;
    # enabling it must not change the one-token decode matrix implementation.
    rows = x.numel() // k
    packed = layer._fl_arm_g32_packed
    if split and rows >= 32:
        packed = layer._fl_arm_g32_prefill
    else:
        split = 0
    out = operators.linear(
        x.reshape(-1, k).float().contiguous(), packed, n, split, workers, sme_workers
    )
    out = out.to(x.dtype).reshape(*x.shape[:-1], n)
    return out if bias is None else out + bias.to(out.dtype)


def install_w8a8():
    from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
        CPUInt8ScaledMMLinearKernel as Kernel,
    )

    if getattr(Kernel, "_fl_arm_native_w8a8", False):
        return False
    original_process = Kernel.process_weights_after_loading
    original_apply = Kernel.apply_weights

    def process(self, layer):
        if not enabled():
            return original_process(self, layer)
        config = self.config
        weight_name, scale_name, *_ = self.layer_param_names
        weight = getattr(layer, weight_name)
        if (
            config.is_static_input_scheme
            or config.input_symmetric
            or not config.is_channelwise
            or weight.ndim != 2
            or weight.dtype != torch.int8
            or weight.shape[1] % 32
        ):
            return original_process(self, layer)
        from vllm_fl.ops.arm_operator_backend import flaggems_ops

        n, k = weight.shape
        scale = getattr(layer, scale_name).detach().reshape(-1).float().contiguous()
        if scale.numel() != n:
            raise ValueError("W8A8 requires one scale per output channel")
        operators = flaggems_ops("w8a8")
        layer._fl_arm_w8_packed = torch.nn.Parameter(
            operators.pack(weight.detach().contiguous(), scale, False),
            requires_grad=False,
        )
        setattr(layer, weight_name, None)
        setattr(layer, scale_name, None)
        self._fl_arm_w8 = (operators, n, k)

    def apply(self, layer, x, bias=None):
        if not hasattr(self, "_fl_arm_w8"):
            return original_apply(self, layer, x, bias)
        operators, n, k = self._fl_arm_w8
        out = operators.linear(
            x.reshape(-1, k).float().contiguous(),
            layer._fl_arm_w8_packed,
            n,
            False,
            32,
            128,
        )
        out = out.to(x.dtype).reshape(*x.shape[:-1], n)
        return out if bias is None else out + bias.to(out.dtype)

    Kernel.process_weights_after_loading = process
    Kernel.apply_weights = apply
    Kernel._fl_arm_native_w8a8 = True
    return True
