# Copyright (c) 2026 BAAI. All rights reserved.

import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

from vllm_fl.ops import cpu_qwen_gdn, cpu_qwen_runtime


class TestCpuQwenRuntime(unittest.TestCase):
    def test_runtime_registration_does_not_set_machine_policy(self):
        q4_module = types.ModuleType("flag_gems.runtime.backend._arm.q4")
        q4_module.enable_vllm_q4_codegen = Mock()
        integration_module = types.ModuleType("flag_gems.integrations.vllm")
        integration_module.maybe_install_kernel_coverage = Mock()
        compatibility = Mock()
        gdn_bridge = Mock()

        compatibility_module = types.ModuleType(
            "vllm_fl.patches.arm_cpu_vllm_0202"
        )
        compatibility_module.install_arm_cpu_vllm_0202_compat = compatibility

        policy_names = {
            "FLAGGEMS_VENDOR",
            "TRITON_CPU_BACKEND",
            "TRITON_LOCAL_LIBOMP_PATH",
            "FLAGGEMS_GDN_TRITON_THREADS",
            "OMP_NUM_THREADS",
        }
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(
                sys.modules,
                {
                    compatibility_module.__name__: compatibility_module,
                    q4_module.__name__: q4_module,
                    integration_module.__name__: integration_module,
                },
            ),
            patch.object(
                cpu_qwen_gdn,
                "install_vllm_gdn_bridge",
                gdn_bridge,
            ),
            patch.object(cpu_qwen_runtime, "_ACTIVE", False),
        ):
            self.assertTrue(cpu_qwen_runtime.enable_qwen_runtime(verbose=False))
            self.assertTrue(policy_names.isdisjoint(os.environ))

        compatibility.assert_called_once_with()
        gdn_bridge.assert_called_once_with()
        q4_module.enable_vllm_q4_codegen.assert_called_once_with(
            verbose=False,
            runtime="libtriton_jit",
        )
        integration_module.maybe_install_kernel_coverage.assert_called_once_with()

    def test_gdn_bridge_delegates_to_flaggems(self):
        integration = types.ModuleType("flag_gems.integrations.vllm")
        integration.install_qwen_gdn = Mock()
        with patch.dict(sys.modules, {integration.__name__: integration}):
            cpu_qwen_gdn.install_vllm_gdn_bridge()
        integration.install_qwen_gdn.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
