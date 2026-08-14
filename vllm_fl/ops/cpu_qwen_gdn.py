"""Thin vLLM bridge to the FlagGems Qwen GDN integration."""

from __future__ import annotations


def install_vllm_gdn_bridge() -> None:
    """Install the version-adapted FlagGems GDN runtime lazily."""
    from flag_gems.integrations.vllm import install_qwen_gdn

    install_qwen_gdn()


__all__ = ["install_vllm_gdn_bridge"]
