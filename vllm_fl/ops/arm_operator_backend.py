# Copyright (c) 2026 BAAI. All rights reserved.
"""Model adapters select FlagGems-owned arithmetic, with no plugin C++ JIT."""

import os


def flaggems_ops(group):
    from flag_gems.runtime.backend._arm.tle_raw.native_ops import arm_native_ops

    backend = os.environ.get("FL_ARM_OPERATOR_BACKEND", "raw")
    if backend not in ("native", "raw"):
        raise ValueError("FL_ARM_OPERATOR_BACKEND must be native or raw")
    return arm_native_ops(group, raw=backend == "raw")
