#!/usr/bin/env python3
"""Measure W4A8 TLE cost inside one compiled multi-Linear graph."""

import argparse
import json
import os
import statistics
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=42)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument(
        "--backend", choices=("plugin", "native"), default="plugin"
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    from vllm_fl.ops import cpu_int4_tleraw as int4

    n = k = args.size
    torch.manual_seed(20260716)
    weight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    scales = torch.rand(n, 1, dtype=torch.bfloat16) * 0.1 + 1e-3
    rhs = int4._pack_rhs(weight, scales)
    unsigned = weight.add(8).to(torch.uint8)
    native_rhs = (
        unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    ).contiguous()
    torch_rhs = torch.ops.aten._dyn_quant_pack_4bit_weight(
        native_rhs, scales.float().contiguous(), None, k, k, n
    )
    x = torch.randn(args.m, k, dtype=torch.bfloat16)

    def chain(value):
        for _ in range(args.layers):
            if args.backend == "plugin":
                value = int4.linear_w4a8(value, rhs, n, k)
            else:
                value = torch.ops.aten._dyn_quant_matmul_4bit(
                    value, torch_rhs, k, k, n
                )
        return value

    compiled = torch.compile(chain, fullgraph=True, dynamic=True)
    compiled(x)
    if os.environ.get("FL_W4A8_PROFILE") == "1":
        int4.profile_reset()
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        for _ in range(args.iterations):
            compiled(x)
        samples.append(
            (time.perf_counter() - start)
            / args.iterations
            / args.layers
            * 1e6
        )
    result = {
        "median_us_per_linear": statistics.median(samples),
        "samples_us_per_linear": samples,
        "gomp_spincount": os.environ.get("GOMP_SPINCOUNT", "default"),
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", "default"),
        "backend": args.backend,
        "m": args.m,
        "size": args.size,
    }
    if os.environ.get("FL_W4A8_PROFILE") == "1":
        result["w4a8_profile"] = int4.profile_stats()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
