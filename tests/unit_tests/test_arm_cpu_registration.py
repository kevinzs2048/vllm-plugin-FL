# Copyright (c) 2026 BAAI. All rights reserved.

import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import vllm_fl


class TestArmCpuRegistration(unittest.TestCase):
    def test_import_does_not_eagerly_import_flaggems(self):
        script = textwrap.dedent(
            """
            import builtins

            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "flag_gems" or name.startswith("flag_gems."):
                    raise RuntimeError("vllm_fl imported FlagGems eagerly")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = guarded_import
            import vllm_fl
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_arm_cpu_build_is_selected(self):
        with (
            patch.object(vllm_fl.platform, "machine", return_value="aarch64"),
            patch.object(vllm_fl.metadata, "version", return_value="0.20.2+cpu"),
        ):
            self.assertTrue(vllm_fl._is_arm_cpu_build())

    def test_arm_accelerator_build_is_not_selected_as_cpu(self):
        with (
            patch.object(vllm_fl.platform, "machine", return_value="aarch64"),
            patch.object(vllm_fl.metadata, "version", return_value="0.20.2"),
        ):
            self.assertFalse(vllm_fl._is_arm_cpu_build())

    def test_non_arm_cpu_build_is_not_selected(self):
        with (
            patch.object(vllm_fl.platform, "machine", return_value="x86_64"),
            patch.object(vllm_fl.metadata, "version", return_value="0.20.2+cpu"),
        ):
            self.assertFalse(vllm_fl._is_arm_cpu_build())

    def test_w4a8_default_requires_flagtree_runtime_sources(self):
        with patch.object(vllm_fl.importlib.util, "find_spec") as find_spec:
            find_spec.return_value = None
            self.assertFalse(vllm_fl._w4a8_assets_configured())

            find_spec.return_value = object()
            self.assertTrue(vllm_fl._w4a8_assets_configured())


if __name__ == "__main__":
    unittest.main()
