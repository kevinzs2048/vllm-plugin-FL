# Copyright (c) 2026 BAAI. All rights reserved.

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
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

    def test_w4a8_default_requires_configured_native_assets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            kai_dir = root / "build"
            kleidiai_root = root / "kleidiai"
            kai_dir.mkdir()
            (kleidiai_root / "kai").mkdir(parents=True)

            with patch.dict(vllm_fl.os.environ, {}, clear=True):
                self.assertFalse(vllm_fl._w4a8_assets_configured())

                (kai_dir / "libkai_w4a8_ukernels.o").touch()
                vllm_fl.os.environ["FL_KAI_W4A8_DIR"] = str(kai_dir)
                vllm_fl.os.environ["KLEIDIAI_ROOT"] = str(kleidiai_root)
                self.assertTrue(vllm_fl._w4a8_assets_configured())


if __name__ == "__main__":
    unittest.main()
