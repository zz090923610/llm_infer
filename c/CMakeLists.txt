cmake_minimum_required(VERSION 3.16)
project(llm_infer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)
add_compile_definitions(_POSIX_C_SOURCE=200809L)
if(ANDROID)
  add_compile_definitions(_GNU_SOURCE)
endif()

get_filename_component(LLM_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# Compute backend: portable scalar CPU today. Add a folder under backends/
# (x86_64, aarch64, npu, gpu) that exports the same kernel symbols.
set(LLM_BACKEND "cpu" CACHE STRING "Compute backend (cpu, x86_64, aarch64, npu, gpu)")
set_property(CACHE LLM_BACKEND PROPERTY STRINGS cpu x86_64 aarch64 npu gpu)

set(LLM_BACKEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/backends/${LLM_BACKEND}")
if(NOT EXISTS "${LLM_BACKEND_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
          "Unknown LLM_BACKEND=${LLM_BACKEND}. Add c/backends/${LLM_BACKEND}/ or use -DLLM_BACKEND=cpu")
endif()
unset(LLM_BACKEND_COMPILE_OPTIONS)
add_subdirectory(backends/${LLM_BACKEND})

set(LLM_SOURCES
    util.c
    hashmap.c
    heap.c
    quant.c
    gguf.c
    unicode.c
    unicode_data.c
    tokenizer.c
    cache.c
    model.c
    sampler.c
    generate.c
    ${LLM_BACKEND_SOURCES}
)

add_library(llm STATIC ${LLM_SOURCES})
target_include_directories(llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llm PUBLIC m)
target_compile_definitions(llm PUBLIC LLM_BACKEND_NAME="${LLM_BACKEND}")
if(LLM_BACKEND_COMPILE_OPTIONS)
  target_compile_options(llm PUBLIC ${LLM_BACKEND_COMPILE_OPTIONS})
endif()
if(LLM_BACKEND STREQUAL "x86_64" OR LLM_BACKEND STREQUAL "aarch64")
  find_package(Threads REQUIRED)
  target_link_libraries(llm PUBLIC Threads::Threads)
endif()
# Host builds bake in the repo GGUF path. Cross/Android binaries must take --model.
if(NOT CMAKE_CROSSCOMPILING AND NOT ANDROID)
  target_compile_definitions(llm PUBLIC LLM_DEFAULT_MODEL="${LLM_REPO_ROOT}/nanogpt-chat-q8_0.gguf")
endif()

add_executable(chat chat.c)
target_link_libraries(chat PRIVATE llm)

add_executable(generate generate_main.c)
target_link_libraries(generate PRIVATE llm)

if(NOT CMAKE_CROSSCOMPILING)
  enable_testing()

  add_executable(test_quant test_quant.c)
  target_link_libraries(test_quant PRIVATE llm)
  add_test(NAME quant COMMAND test_quant)

  add_executable(test_tokenizer test_tokenizer.c)
  target_link_libraries(test_tokenizer PRIVATE llm)
  add_test(NAME tokenizer COMMAND test_tokenizer)

  if(LLM_BACKEND STREQUAL "x86_64" OR LLM_BACKEND STREQUAL "aarch64")
    add_executable(test_linear test_linear.c)
    target_link_libraries(test_linear PRIVATE llm)
    add_test(NAME linear COMMAND test_linear)
  endif()
endif()
