#!/usr/bin/env bash
# Cross-compile chat/generate for Android arm64-v8a (Bionic).
# Requires: cmake, NDK at ANDROID_NDK or ANDROID_NDK_HOME.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/c/build-android}"
NDK="${ANDROID_NDK:-${ANDROID_NDK_HOME:-}}"
# Default: NDK r27d (smallest current LTS with 16 KiB page support for Ace 5).
if [[ -z "$NDK" && -f "$HOME/Android/Sdk/ndk/27.3.13750724/build/cmake/android.toolchain.cmake" ]]; then
  NDK="$HOME/Android/Sdk/ndk/27.3.13750724"
fi

if [[ -z "$NDK" ]]; then
  echo "error: set ANDROID_NDK or ANDROID_NDK_HOME to your NDK path" >&2
  exit 1
fi
if [[ ! -f "$NDK/build/cmake/android.toolchain.cmake" ]]; then
  echo "error: $NDK is not an Android NDK (missing build/cmake/android.toolchain.cmake)" >&2
  exit 1
fi

ABI="${ANDROID_ABI:-arm64-v8a}"
PLATFORM="${ANDROID_PLATFORM:-android-28}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

# NEON + pthread backend for Snapdragon 8 Gen 3 (OnePlus Ace 5).
# GPU: LLM_BACKEND=gpu (Vulkan / Adreno 750).
BACKEND="${LLM_BACKEND:-aarch64}"

cmake -S "$ROOT/c" -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ABI" \
  -DANDROID_PLATFORM="$PLATFORM" \
  -DANDROID_STL=none \
  -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLM_BACKEND="$BACKEND"

cmake --build "$BUILD_DIR" -j"$JOBS"

echo
for bin in chat generate; do
  path="$BUILD_DIR/$bin"
  if [[ ! -f "$path" ]]; then
    echo "error: missing $path" >&2
    exit 1
  fi
  if command -v file >/dev/null 2>&1; then
    file "$path"
  fi
  if command -v readelf >/dev/null 2>&1; then
    readelf -h "$path" | grep -E 'Class:|Machine:|Flags:'
  fi
done
echo "Android $ABI binaries: $BUILD_DIR/{chat,generate}"
if [[ -f "$BUILD_DIR/test_linear" ]]; then
  echo "  plus $BUILD_DIR/test_linear"
fi
if [[ "$BACKEND" == "gpu" ]]; then
  echo "GPU binaries need vendor ICD libs on device:"
  echo "  export LD_LIBRARY_PATH=/vendor/lib64:/vendor/lib64/hw"
fi
