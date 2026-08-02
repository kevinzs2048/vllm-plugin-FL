# Copyright (c) 2025 BAAI. All rights reserved.

import logging
import os
import pathlib
import platform
import sys
from importlib import metadata

# torch.float4_e2m1fn_x2 exists only in CUDA builds of PyTorch 2.7+.
# vllm.ir.tolerances references it at module level, so we inject a sentinel
# before any vllm.ir import can happen.
if "torch" in sys.modules:
    _torch = sys.modules["torch"]
    if not hasattr(_torch, "float4_e2m1fn_x2"):
        _torch.float4_e2m1fn_x2 = _torch.uint8
else:
    import torch as _torch
    if not hasattr(_torch, "float4_e2m1fn_x2"):
        _torch.float4_e2m1fn_x2 = _torch.uint8
del _torch

from vllm_fl.utils import get_op_config as _get_op_config

from . import version as version  # PyTorch-style: vllm_fl.version.git_version


logger = logging.getLogger(__name__)


def _is_arm_cpu_build() -> bool:
    """Return whether this is an AArch64 vLLM CPU build.

    Architecture alone is insufficient: accelerator-enabled vLLM wheels also
    run on AArch64 hosts and must retain the original FL platform.
    """
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        return False
    try:
        return "cpu" in metadata.version("vllm").lower()
    except metadata.PackageNotFoundError:
        return os.environ.get("VLLM_TARGET_DEVICE", "").lower() == "cpu"


def _w4a8_assets_configured() -> bool:
    """Return whether the external W4A8 runtime inputs are usable."""
    kai_dir = os.environ.get("FL_KAI_W4A8_DIR", "")
    if not kai_dir or not (
        pathlib.Path(kai_dir) / "libkai_w4a8_ukernels.o"
    ).is_file():
        return False
    pack_library = os.environ.get("FL_KAI_W4A8_PACK_SO", "")
    if pack_library:
        return pathlib.Path(pack_library).is_file()
    kleidiai_root = os.environ.get("KLEIDIAI_ROOT", "")
    return bool(kleidiai_root) and (pathlib.Path(kleidiai_root) / "kai").is_dir()


def __getattr__(name):
    if name == "distributed":
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _patch_transformers_compat():
    """Patch transformers compatibility for ALLOWED_LAYER_TYPES and tokenizer."""
    import transformers.configuration_utils as cfg
    if not hasattr(cfg, "ALLOWED_LAYER_TYPES"):
        cfg.ALLOWED_LAYER_TYPES = getattr(
            cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ()
        )


def _register_flagcx_connector():
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    for _alias in ("FlagCXConnector", "FlagcxConnector"):
        if _alias not in KVConnectorFactory._registry:
            KVConnectorFactory.register_connector(
                _alias,
                "vllm_fl.distributed.kv_transfer.flagcx_connector",
                "FlagCXConnector",
            )


def _patch_flash_attn_import():
    """Stub vllm.vllm_flash_attn if CUDA flash attention C extensions are missing."""
    import sys
    if "vllm.vllm_flash_attn" in sys.modules:
        return
    try:
        import vllm.vllm_flash_attn  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("vllm.vllm_flash_attn")
        stub.FA2_AVAILABLE = False
        stub.FA3_AVAILABLE = False
        stub.fa_version_unsupported_reason = lambda *a, **kw: "flash_attn C extensions not available"
        stub.flash_attn_varlen_func = None
        stub.get_scheduler_metadata = None
        stub.is_fa_version_supported = lambda *a, **kw: False
        sys.modules["vllm.vllm_flash_attn"] = stub


def _patch_custom_ops():
    """Register torch.ops._C op schemas when vllm._C is unavailable."""
    try:
        import vllm._C  # noqa: F401
        return
    except (ImportError, OSError):
        pass

    try:
        import vllm_fl._C  # noqa: F401
    except (ImportError, OSError) as e:
        logger.debug("Failed to import vllm_fl._C: %s", e)

    from vllm_fl.ops._C_ops_registry import register_op_schemas
    register_op_schemas()


def register():
    """Register the FL platform."""
    # PlatformFL is accelerator-shaped. ARM CPU uses vLLM's native CPU platform
    # plus the FL TLE-raw W4A8 integration installed by register_model().
    if _is_arm_cpu_build():
        logger.info("[vllm_fl] ARM CPU -> FL CPU platform (native-backed)")
        return "vllm_fl.platform_cpu.CpuPlatformFL"

    _patch_custom_ops()
    _patch_flash_attn_import()
    _patch_transformers_compat()

    # Model-specific platform patches
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform
    glm5_platform()

    # Note: FlagCX connector registration is deferred to register_model()
    # to avoid circular imports during VllmConfig.__post_init__ in spawned
    # subprocesses.

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()

    return "vllm_fl.platform.PlatformFL"

def register_quant_linear():
    from vllm.platforms import current_platform
    # vllm.model_executor.kernels.linear triggers cutlass_scaled_mm_supports_fp8
    # at module level, which requires torch.ops._C — not available on MUSA.
    if current_platform.device_type == "musa":
        return
    from vllm_fl.quantization.quant_linear import add_oot_quant_kernel
    add_oot_quant_kernel()

def register_router():
    from vllm.platforms import current_platform
    # fused_moe import chain triggers cutlass_scaled_mm_supports_fp8 on MUSA
    if current_platform.device_type == "musa":
        return
    from vllm_fl.utils import is_oot_enabled
    if not is_oot_enabled():
        return
    from vllm_fl.ops.fused_moe.router import replace_router_with_fl
    replace_router_with_fl()

def register_model():
    """Register FL-specific models not yet upstream."""
    from vllm.platforms import current_platform
    if current_platform.device_type == "cpu" and _is_arm_cpu_build():
        # INT8 modes have priority when explicitly enabled. KleidiAI/TLE-raw
        # are W8A8 dynamic; torchpack is the torch-native W8A16 fallback.
        int8_enabled = os.environ.get("FL_CPU_INT8", "0").lower()
        if int8_enabled not in {"0", "1", "false", "true"}:
            raise ValueError("FL_CPU_INT8 must be one of: 0, 1, false, true")
        if int8_enabled in {"1", "true"}:
            int8_backend = os.environ.get(
                "FL_CPU_INT8_BACKEND", "tleraw"
            ).lower()
            if int8_backend == "tleraw":
                from vllm_fl.ops.cpu_int8_tleraw import enable_int8
            elif int8_backend == "kleidiai":
                from vllm_fl.ops.cpu_int8_kai import enable_int8
            elif int8_backend == "torchpack":
                from vllm_fl.ops.cpu_int8_pack import enable_int8
            else:
                raise ValueError(
                    "FL_CPU_INT8_BACKEND must be 'tleraw', 'kleidiai' or 'torchpack'"
                )

            enable_int8()
            return
        int4_setting = os.environ.get("FL_CPU_INT4")
        if int4_setting is None:
            enabled = "1" if _w4a8_assets_configured() else "0"
            if enabled == "0":
                logger.info(
                    "[vllm_fl] W4A8 assets are not configured -> bf16; "
                    "set FL_CPU_INT4=1 for strict activation"
                )
        else:
            enabled = int4_setting.lower()
            if enabled not in {"0", "1", "false", "true"}:
                raise ValueError("FL_CPU_INT4 must be one of: 0, 1, false, true")
        if enabled in {"1", "true"}:
            backend = os.environ.get("FL_CPU_INT4_BACKEND", "tleraw").lower()
            if backend != "tleraw":
                raise ValueError("FL_CPU_INT4_BACKEND must be 'tleraw'")
            from vllm_fl.ops.cpu_int4_tleraw import enable_int4

            enable_int4()
        else:
            logger.info("[vllm_fl] FL_CPU_INT4=0 -> bf16 (int4 op skipped)")
        return

    _register_flagcx_connector()

    # Register OOT quant kernels so kernel selection can find them
    register_quant_linear()
    register_router()

    # Register GLM-5 (GlmMoeDsa) — config not yet upstream
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig
        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        #from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        #glm5_model()
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")
