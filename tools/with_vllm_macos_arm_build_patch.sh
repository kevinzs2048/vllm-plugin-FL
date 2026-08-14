#!/usr/bin/env bash
set -eu

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <vllm-source-root> <build-command> [args ...]" >&2
  exit 2
fi

VLLM_SOURCE_ROOT=$1
shift
EXPECTED_COMMIT=bc150f50299199599673614f80d12a196f377655
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PATCH_FILE=$SCRIPT_DIR/patches/vllm-0.20.2-macos-arm-attention.patch

actual_commit=$(git -C "$VLLM_SOURCE_ROOT" rev-parse HEAD)
if [[ $actual_commit != "$EXPECTED_COMMIT" ]]; then
  echo "expected clean vLLM 0.20.2 at $EXPECTED_COMMIT, found $actual_commit" >&2
  exit 1
fi
if ! git -C "$VLLM_SOURCE_ROOT" diff --quiet \
  || ! git -C "$VLLM_SOURCE_ROOT" diff --cached --quiet; then
  echo "vLLM source tree must be clean before applying the build patch" >&2
  exit 1
fi

git -C "$VLLM_SOURCE_ROOT" apply --check "$PATCH_FILE"
git -C "$VLLM_SOURCE_ROOT" apply "$PATCH_FILE"
cleanup() {
  git -C "$VLLM_SOURCE_ROOT" apply --reverse "$PATCH_FILE"
}
trap cleanup EXIT INT TERM

(cd "$VLLM_SOURCE_ROOT" && "$@")
