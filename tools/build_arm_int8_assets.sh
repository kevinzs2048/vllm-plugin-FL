#!/usr/bin/env bash
# Build libkai_w8a8.so: KleidiAI qai8dxp x qsi8cxp ukernels + FL wrapper.
# usage: build_arm_int8_assets.sh [KLEIDIAI_ROOT]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OPS_DIR=$(cd "${HERE}/../vllm_fl/ops" && pwd)

KLEIDIAI_ROOT=${1:-${KLEIDIAI_ROOT:-}}
if [[ -z ${KLEIDIAI_ROOT} || ! -d ${KLEIDIAI_ROOT}/kai ]]; then
    echo "pass a KleidiAI source root or set KLEIDIAI_ROOT" >&2
    exit 1
fi

CFLAGS=(-O3 -fPIC -fopenmp -march=armv8.6-a+bf16+i8mm+dotprod -I"${KLEIDIAI_ROOT}")
MATMUL_DIR="${KLEIDIAI_ROOT}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp"
PACK_DIR="${KLEIDIAI_ROOT}/kai/ukernels/matmul/pack"

SRCS=(
    "${MATMUL_DIR}/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c"
    "${MATMUL_DIR}/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c"
    "${PACK_DIR}/kai_lhs_quant_pack_qai8dxp_bf16_neon.c"
    "${PACK_DIR}/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c"
    "${OPS_DIR}/cpu_int8_kai_wrapper.c"
)

gcc "${CFLAGS[@]}" -shared -o "${OPS_DIR}/libkai_w8a8.so" "${SRCS[@]}" -lm
echo "built ${OPS_DIR}/libkai_w8a8.so"
