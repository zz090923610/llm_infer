#!/usr/bin/env bash
# Push Android arm64 binaries and the GGUF to the phone Downloads folder
# (visible in Files as Download/llm_infer). /sdcard is FUSE+noexec, so run
# the same files via /mnt/pass_through/... instead of ./chat or linker64.
# Usage: adb-push.sh [--run] [--model PATH] [--build-dir DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$ROOT/c/build-android"
MODEL="$ROOT/models/smollm2-360m-instruct-q8_0.gguf"
REMOTE="/sdcard/Download/llm_infer"
# Same directory as $REMOTE, but on the f2fs pass-through. /sdcard is FUSE+noexec,
# so ./chat and linker64 both fail there (Permission denied / map segment EPERM).
RUN_DIR="/mnt/pass_through/0/emulated/0/Download/llm_infer"
RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--run] [--model PATH] [--build-dir DIR]"
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v adb >/dev/null 2>&1; then
  echo "error: adb not found on PATH" >&2
  exit 1
fi

CHAT="$BUILD_DIR/chat"
GEN="$BUILD_DIR/generate"
TEST_LINEAR="$BUILD_DIR/test_linear"
if [[ ! -x "$CHAT" && ! -f "$CHAT" ]]; then
  echo "error: missing $CHAT (run c/scripts/build-android.sh first)" >&2
  exit 1
fi
if [[ ! -f "$GEN" ]]; then
  echo "error: missing $GEN (run c/scripts/build-android.sh first)" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: missing model $MODEL" >&2
  exit 1
fi

MODEL_NAME="$(basename "$MODEL")"

echo "pushing to $REMOTE (Files app: Download/llm_infer)"
adb shell "mkdir -p '$REMOTE'"
adb push "$CHAT" "$REMOTE/chat"
adb push "$GEN" "$REMOTE/generate"
if [[ -f "$TEST_LINEAR" ]]; then
  adb push "$TEST_LINEAR" "$REMOTE/test_linear"
fi
adb push "$MODEL" "$REMOTE/$MODEL_NAME"
if [[ -f "$TEST_LINEAR" ]]; then
  adb shell "chmod 755 '$RUN_DIR/chat' '$RUN_DIR/generate' '$RUN_DIR/test_linear'"
else
  adb shell "chmod 755 '$RUN_DIR/chat' '$RUN_DIR/generate'"
fi

echo "on device (run via pass-through; /sdcard is FUSE noexec):"
echo "  adb shell -t"
echo "  export LD_LIBRARY_PATH=/vendor/lib64:/vendor/lib64/hw"
echo "  $RUN_DIR/generate --model $RUN_DIR/$MODEL_NAME --prompt \"Hello\" --temp 0 --max-tokens 32"
echo "  $RUN_DIR/chat --model $RUN_DIR/$MODEL_NAME --ctx 1024"
if [[ -f "$TEST_LINEAR" ]]; then
  echo "  $RUN_DIR/test_linear"
fi

if [[ "$RUN" -eq 1 ]]; then
  GPU_ENV="LD_LIBRARY_PATH=/vendor/lib64:/vendor/lib64/hw"
  if [[ -f "$TEST_LINEAR" ]]; then
    echo
    echo "smoke: test_linear"
    adb shell "$GPU_ENV $RUN_DIR/test_linear"
  fi
  echo
  echo "smoke: generate --prompt Hello --temp 0 --max-tokens 16"
  adb shell "$GPU_ENV $RUN_DIR/generate --model '$RUN_DIR/$MODEL_NAME' --prompt Hello --temp 0 --max-tokens 16"
fi
