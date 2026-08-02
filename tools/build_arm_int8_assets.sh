#!/usr/bin/env bash
# Build the sole packaged W8A8 native asset: KleidiAI ukernels + FL wrapper.
# usage: build_arm_int8_assets.sh [KLEIDIAI_ROOT] [OUTPUT_DIR]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OPS_DIR=$(cd "${HERE}/../vllm_fl/ops" && pwd)

KLEIDIAI_ROOT=${1:-${KLEIDIAI_ROOT:-}}
if [[ -z ${KLEIDIAI_ROOT} || ! -d ${KLEIDIAI_ROOT}/kai ]]; then
    echo "pass a KleidiAI source root or set KLEIDIAI_ROOT" >&2
    exit 1
fi
OUTPUT_DIR=${2:-${FL_KAI_W8A8_OUTPUT_DIR:-${OPS_DIR}}}
mkdir -p "${OUTPUT_DIR}"
CC=${CC:-gcc}

CFLAGS=(-O3 -fPIC -fvisibility=hidden -fopenmp -march=armv8.6-a+bf16+i8mm+dotprod -I"${KLEIDIAI_ROOT}")
MATMUL_DIR="${KLEIDIAI_ROOT}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp"
PACK_DIR="${KLEIDIAI_ROOT}/kai/ukernels/matmul/pack"

SRCS=(
    "${MATMUL_DIR}/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c"
    "${MATMUL_DIR}/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c"
    "${PACK_DIR}/kai_lhs_quant_pack_qai8dxp_bf16_neon.c"
    "${PACK_DIR}/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c"
    "${OPS_DIR}/cpu_int8_kai_wrapper.c"
)

OUTPUT_LIBRARY="${OUTPUT_DIR}/libkai_w8a8.so"
TEMP_LIBRARY=$(mktemp "${OUTPUT_DIR}/.libkai_w8a8.so.XXXXXX")
trap 'rm -f "${TEMP_LIBRARY}"' EXIT
"${CC}" "${CFLAGS[@]}" -shared -o "${TEMP_LIBRARY}" "${SRCS[@]}" -lm
chmod 0755 "${TEMP_LIBRARY}"
mv "${TEMP_LIBRARY}" "${OUTPUT_LIBRARY}"
trap - EXIT
echo "built ${OUTPUT_DIR}/libkai_w8a8.so"
if git -C "${KLEIDIAI_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
    echo "KleidiAI revision: $(git -C "${KLEIDIAI_ROOT}" rev-parse HEAD)"
fi
