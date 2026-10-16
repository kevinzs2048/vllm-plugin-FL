#!/bin/bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORK_ROOT="${WORK_ROOT:-$(cd "$HERE/.." && pwd)}"
mkdir -p "$WORK_ROOT/source"
fetch() {
  local name="$1" url="$2" commit="$3" directory="$WORK_ROOT/source/$1"
  local branch="${4:-minicpm-v47-arm-triton-tle-20261011}"
  if [ ! -e "$directory/.git" ]; then
    git clone --filter=blob:none --branch "$branch" "$url" "$directory"
  fi
  if [ -n "$(git -C "$directory" status --porcelain)" ]; then
    printf 'Refusing to replace changed source: %s\n' "$directory" >&2
    exit 1
  fi
  git -C "$directory" fetch origin "$commit"
  git -C "$directory" checkout --detach "$commit"
  test "$(git -C "$directory" rev-parse HEAD)" = "$commit"
}
fetch FlagGems https://github.com/kevinzs2048/FlagGems.git b5f5e110c7c2ae40521d0de897f3961467265fa5 minicpm-v47-arm-nokai-20261015
fetch flagtree-cpu https://github.com/kevinzs2048/flagtree-cpu.git 36133df9540cdd9eb5b09209ee41ce1697b27126
fetch vllm-plugin-FL https://github.com/kevinzs2048/vllm-plugin-FL.git f0de702335902facfffe625c0690f7a3df7f617d
# Recursive submodules are required only for optional compiler source builds.
if [ "${FETCH_BUILD_SUBMODULES:-0}" = 1 ]; then
  git -C "$WORK_ROOT/source/flagtree-cpu" submodule update --init --recursive
  test "$(git -C "$WORK_ROOT/source/flagtree-cpu/third_party/sleef" rev-parse HEAD)" = 93f04d869471ce4d007abaebb8c6a7bc62749f61
fi
