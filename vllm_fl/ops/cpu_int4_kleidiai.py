"""Compatibility alias for the former ctypes ARM INT4 prototype.

Runtime compute now exclusively uses the FlagTree TLE-raw integration. Keep
this module so existing local imports continue to enable the supported path.
"""

from vllm_fl.ops.cpu_int4_tleraw import enable_int4, stats

__all__ = ["enable_int4", "stats"]
