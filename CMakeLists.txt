cmake_minimum_required(VERSION 3.16)
project(llm_infer C)

# Host build lives at the repo root: `cmake -B build && cmake --build build -j`.
# Binaries (`generate-plain-cpu`, `chat-x86_64-simd`, …) land in this top-level build dir,
# not under build/c/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

enable_testing()
add_subdirectory(c)
