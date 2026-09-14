if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  message(FATAL_ERROR
          "LLM_BACKEND=aarch64-simd requires AArch64 (got ${CMAKE_SYSTEM_PROCESSOR}). "
          "Cross-compile with the Android NDK, or use -DLLM_BACKEND=plain-cpu")
endif()

# NEON + FMA are mandatory on AArch64. armv8.2-a matches Snapdragon 8 Gen 3
# (OnePlus Ace 5) and is still accepted by the NDK android-28 clang.
set(LLM_BACKEND_COMPILE_OPTIONS -march=armv8.2-a PARENT_SCOPE)

set(LLM_BACKEND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/backend.c
    ${CMAKE_CURRENT_SOURCE_DIR}/pool.c
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../pc/plain-cpu/rope.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../pc/plain-cpu/attn.c
    PARENT_SCOPE
)
