# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniCPM-V 4.7 configs/processors and lazy vLLM model registration."""
import threading

_LOCK = threading.Lock()
_ACTIVE = False


def register_model():
    """Register both dense and MoE models without initializing CPU kernels."""
    global _ACTIVE
    with _LOCK:
        if _ACTIVE:
            return False
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__).release[:2] < (5,7):
            # This optional model's vendored config/processor APIs require
            # Transformers 5.7. Other platform/model registrations continue.
            return False
        from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoVideoProcessor
        from vllm import ModelRegistry
        from vllm.transformers_utils.config import _CONFIG_REGISTRY

        from .configuration import MiniCPMV4_7Config
        from .hf.configuration_minicpmv4_7 import MiniCPMV4_7VisionConfig
        from .hf.image_processing_minicpmv4_7 import MiniCPMV4_7ImageProcessor
        from .hf.image_processing_pil_minicpmv4_7 import MiniCPMV4_7ImageProcessorPil
        from .hf.processing_minicpmv4_7 import MiniCPMV4_7Processor
        from .hf.video_processing_minicpmv4_7 import MiniCPMV4_7VideoProcessor

        # ProcessorMixin resolves serialized component names in Transformers.
        # Publish only these model-specific names; do not change shared video
        # documentation constants or import HF model arithmetic.
        for cls in (MiniCPMV4_7Config, MiniCPMV4_7VisionConfig,
                    MiniCPMV4_7ImageProcessor, MiniCPMV4_7ImageProcessorPil,
                    MiniCPMV4_7VideoProcessor, MiniCPMV4_7Processor):
            setattr(transformers, cls.__name__, cls)
        AutoConfig.register('minicpmv4_7', MiniCPMV4_7Config, exist_ok=True)
        AutoConfig.register('minicpmv4_7_vision', MiniCPMV4_7VisionConfig, exist_ok=True)
        _CONFIG_REGISTRY['minicpmv4_7'] = MiniCPMV4_7Config
        AutoImageProcessor.register(MiniCPMV4_7Config,
            image_processor_classes={'torchvision': MiniCPMV4_7ImageProcessor,
                                     'pil': MiniCPMV4_7ImageProcessorPil}, exist_ok=True)
        AutoVideoProcessor.register(MiniCPMV4_7Config, MiniCPMV4_7VideoProcessor, exist_ok=True)
        AutoProcessor.register(MiniCPMV4_7Config, MiniCPMV4_7Processor, exist_ok=True)
        for name in ('MiniCPMV4_7ForConditionalGeneration',
                     'MiniCPMV4_7DenseForConditionalGeneration'):
            ModelRegistry.register_model(name, f'vllm_fl.models.minicpmv4_7.model:{name}')
        _ACTIVE = True
        return True


def worker_options():
    """Optional explicit CPU worker, leaving vLLM's default unchanged."""
    from vllm_fl.worker.cpu_thread_policy import ThreadPolicy
    if ThreadPolicy.from_env() is not None:
        return {'worker_cls': 'vllm_fl.worker.cpu_phase_worker.CPUPhaseWorker'}
    return {}
