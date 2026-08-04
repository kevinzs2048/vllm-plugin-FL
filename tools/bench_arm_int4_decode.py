#!/usr/bin/env python3
"""Fast A/B benchmark for ARM W4A8 decode GEMV scheduling."""

import argparse
import json
import os
import statistics
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default="1", help="comma-separated M values")
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--kernel",
        choices=("plugin", "native", "all"),
        default="plugin",
    )
    args = parser.parse_args()
    m_values = [int(value) for value in args.m.split(",")]
    if not m_values or any(value < 1 for value in m_values):
        raise ValueError("--m values must be positive")

    from vllm_fl.ops import cpu_int4_tleraw as int4

    torch.manual_seed(20260716)
    weight = torch.randint(-8, 8, (args.n, args.k), dtype=torch.int8)
    scales = torch.rand(args.n, 1, dtype=torch.bfloat16) * 0.1 + 1e-3
    rhs = int4._pack_rhs(weight, scales)
    unsigned = weight.add(8).to(torch.uint8)
    native_rhs = (
        unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    ).contiguous()
    torch_rhs = torch.ops.aten._dyn_quant_pack_4bit_weight(
        native_rhs, scales.float().contiguous(), None, args.k, args.k, args.n
    )

    def plugin_linear(value):
        return int4.linear_w4a8(value, rhs, args.n, args.k)

    def native_linear(value):
        return torch.ops.aten._dyn_quant_matmul_4bit(
            value, torch_rhs, args.k, args.k, args.n
        )

    variants = ("plugin", "native") if args.kernel == "all" else (args.kernel,)
    results = {}
    for m in m_values:
        x = torch.randn(m, args.k, dtype=torch.bfloat16)
        by_kernel = {}
        outputs = {}
        for variant in variants:
            linear = native_linear if variant == "native" else plugin_linear
            for _ in range(10):
                linear(x)
            if variant != "native" and os.environ.get("FL_W4A8_PROFILE") == "1":
                int4.profile_reset()

            samples = []
            output = None
            for _ in range(args.repeats):
                start = time.perf_counter()
                for _ in range(args.iterations):
                    output = linear(x)
                elapsed = time.perf_counter() - start
                samples.append(elapsed / args.iterations * 1e6)
            assert output is not None
            outputs[variant] = output
            by_kernel[variant] = {
                "median_us": statistics.median(samples),
                "samples_us": samples,
                "checksum": float(output.float().sum()),
            }
            if variant != "native" and os.environ.get("FL_W4A8_PROFILE") == "1":
                by_kernel[variant]["profile"] = int4.profile_stats()
        if args.kernel == "all":
            torch.testing.assert_close(
                outputs["plugin"], outputs["native"], rtol=0, atol=0
            )
        results[str(m)] = by_kernel

    print(
        json.dumps(
            {
                "threads": torch.get_num_threads(),
                "nk": [args.n, args.k],
                "results": results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
