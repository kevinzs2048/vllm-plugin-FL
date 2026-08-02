# Copyright (c) 2026 BAAI. All rights reserved.

import unittest
from unittest.mock import patch

import vllm_fl


class TestArmCpuRegistration(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
