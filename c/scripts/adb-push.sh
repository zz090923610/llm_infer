#!/usr/bin/env bash
# Push Android arm64 binaries and the GGUF to the phone Downloads folder
# (visible in Files as Download/llm_infer). Shared storage is usually noexec,
# so run via /system/bin/linker64 rather than ./generate.
# Usage: adb-push.sh [--run] [--model PATH] [--build-dir DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$ROOT/c/build-android"
MODEL="$ROOT/nanogpt-chat-q8_0.gguf"
REMOTE="/sdcard/Download/llm_infer"
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
LINKER="/system/bin/linker64"

echo "pushing to $REMOTE (Files app: Download/llm_infer)"
adb shell "mkdir -p '$REMOTE'"
adb push "$CHAT" "$REMOTE/chat"
adb push "$GEN" "$REMOTE/generate"
adb push "$MODEL" "$REMOTE/$MODEL_NAME"

echo "on device (sdcard is usually noexec; use linker64):"
echo "  adb shell -t"
echo "  cd $REMOTE"
echo "  $LINKER ./generate --model ./$MODEL_NAME --prompt \"Hello\" --temp 0 --max-tokens 32"
echo "  $LINKER ./chat --model ./$MODEL_NAME --ctx 1024"

if [[ "$RUN" -eq 1 ]]; then
  echo
  echo "smoke: generate --prompt Hello --temp 0 --max-tokens 16"
  adb shell "cd '$REMOTE' && $LINKER ./generate --model './$MODEL_NAME' --prompt Hello --temp 0 --max-tokens 16"
fi
