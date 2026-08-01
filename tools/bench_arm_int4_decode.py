#!/usr/bin/env python3
"""Fast A/B benchmark for ARM W4A8 decode GEMV scheduling."""

import argparse
import json
import statistics
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    from vllm_fl.ops import cpu_int4_tleraw as int4

    torch.manual_seed(20260716)
    weight = torch.randn(args.n, args.k, dtype=torch.bfloat16)
    native, scales = int4.quant_native_qs4c32(weight)
    rhs = int4._pack_rhs(native, scales, args.n, args.k)
    x = torch.randn(1, args.k, dtype=torch.bfloat16)
    for _ in range(10):
        int4.linear_w4a8(x, rhs, args.n, args.k)

    samples = []
    output = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        for _ in range(args.iterations):
            output = int4.linear_w4a8(x, rhs, args.n, args.k)
        samples.append((time.perf_counter() - start) / args.iterations * 1e6)

    assert output is not None
    print(
        json.dumps(
            {
                "threads": torch.get_num_threads(),
                "shape": [1, args.k, args.n],
                "median_us": statistics.median(samples),
                "samples_us": samples,
                "checksum": float(output.float().sum()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
