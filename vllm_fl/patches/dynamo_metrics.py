"""Compatibility for Torch serializing FlagGems' logging-function set."""


def patch_dynamo_metrics_serialization() -> None:
    import torch._dynamo.utils as dynamo_utils

    if getattr(dynamo_utils, "_fl_metrics_serialization_patched", False):
        return
    original = dynamo_utils._get_dynamo_config_for_logging
    try:
        original()
    except TypeError:

        def get_dynamo_config_for_logging():
            try:
                return original()
            except TypeError:
                return None

        dynamo_utils._get_dynamo_config_for_logging = get_dynamo_config_for_logging
        dynamo_utils._fl_metrics_serialization_patched = True
