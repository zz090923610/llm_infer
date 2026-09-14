cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(llm_infer C)
  # Standalone `cmake -B build -S c` still dumps binaries in that build dir.
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

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
set(LLM_C_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# "all" builds every backend this host can compile. A single name keeps the
# old one-lib layout (unsuffixed chat/generate) for Android / existing scripts.
set(LLM_BACKEND "all" CACHE STRING
    "Compute backend: all, or one of cpu, x86_64, aarch64, gpu, pim")
set_property(CACHE LLM_BACKEND PROPERTY STRINGS all cpu x86_64 aarch64 gpu pim)

set(LLM_CORE_SOURCES
    util.c
    hashmap.c
    heap.c
    quant.c
    weight.c
    gguf.c
    unicode.c
    unicode_data.c
    tokenizer.c
    cache.c
    gdn.c
    rope_neox.c
    model.c
    sampler.c
    generate.c
)

function(llm_find_pim_func out_var)
  set(_dir "")
  foreach(_cand
          "${LLM_C_DIR}/../../pim_func"
          "${LLM_C_DIR}/../pim_func"
          "${LLM_C_DIR}/../../gem5_se/pim_func")
    get_filename_component(_abs "${_cand}" ABSOLUTE)
    if(EXISTS "${_abs}/CMakeLists.txt")
      set(_dir "${_abs}")
      break()
    endif()
  endforeach()
  set(${out_var} "${_dir}" PARENT_SCOPE)
endfunction()

function(llm_available_backends out_var)
  set(_be cpu)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    list(APPEND _be x86_64)
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    list(APPEND _be aarch64)
  endif()
  llm_find_pim_func(_pim)
  if(_pim)
    list(APPEND _be pim)
  endif()
  find_library(_vk vulkan QUIET)
  find_path(_vk_inc vulkan/vulkan.h QUIET)
  if(_vk AND _vk_inc)
    list(APPEND _be gpu)
  endif()
  set(${out_var} "${_be}" PARENT_SCOPE)
endfunction()

function(llm_add_backend BE SUFFIX)
  unset(LLM_BACKEND_SOURCES)
  unset(LLM_BACKEND_LIBS)
  unset(LLM_BACKEND_INCLUDE)
  unset(LLM_BACKEND_COMPILE_OPTIONS)
  add_subdirectory("${LLM_C_DIR}/backends/${BE}"
                   "${CMAKE_CURRENT_BINARY_DIR}/be_${BE}")

  if(SUFFIX)
    set(_lib llm_${BE})
    set(_chat chat_${BE})
    set(_gen generate_${BE})
    set(_chat_out chat-${BE})
    set(_gen_out generate-${BE})
  else()
    set(_lib llm)
    set(_chat chat)
    set(_gen generate)
    set(_chat_out chat)
    set(_gen_out generate)
  endif()

  add_library(${_lib} STATIC ${LLM_CORE_SOURCES} ${LLM_BACKEND_SOURCES})
  target_include_directories(${_lib} PUBLIC "${LLM_C_DIR}/include")
  target_link_libraries(${_lib} PUBLIC m)
  target_compile_definitions(${_lib} PUBLIC LLM_BACKEND_NAME="${BE}")
  if(LLM_BACKEND_COMPILE_OPTIONS)
    target_compile_options(${_lib} PRIVATE ${LLM_BACKEND_COMPILE_OPTIONS})
  endif()
  if(NOT CMAKE_CROSSCOMPILING AND NOT ANDROID)
    target_compile_definitions(${_lib} PUBLIC
      LLM_DEFAULT_MODEL="${LLM_REPO_ROOT}/models/smollm2-360m-instruct-q8_0.gguf")
  endif()

  if(BE STREQUAL "x86_64" OR BE STREQUAL "aarch64" OR BE STREQUAL "pim" OR
     (BE STREQUAL "gpu" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64"))
    find_package(Threads REQUIRED)
    target_link_libraries(${_lib} PUBLIC Threads::Threads)
    if(NOT BE STREQUAL "pim")
      target_compile_definitions(${_lib} PRIVATE LLM_HAS_POOL)
    endif()
  endif()
  if(BE STREQUAL "x86_64")
    target_include_directories(${_lib} PRIVATE "${LLM_C_DIR}/backends/x86_64")
  endif()
  if(BE STREQUAL "pim" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_include_directories(${_lib} PRIVATE "${LLM_C_DIR}/backends/x86_64")
    target_compile_definitions(${_lib} PRIVATE PIM_USE_X86_CPU_OPS LLM_HAS_POOL)
  endif()
  if(BE STREQUAL "aarch64" OR
     (BE STREQUAL "gpu" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64"))
    target_include_directories(${_lib} PRIVATE "${LLM_C_DIR}/backends/aarch64")
  endif()
  if(LLM_BACKEND_INCLUDE)
    target_include_directories(${_lib} PRIVATE ${LLM_BACKEND_INCLUDE})
  endif()
  if(LLM_BACKEND_LIBS)
    target_link_libraries(${_lib} PUBLIC ${LLM_BACKEND_LIBS})
  endif()
  if(BE STREQUAL "pim")
    target_include_directories(${_lib} PRIVATE "${LLM_C_DIR}/backends/pim")
    set_target_properties(${_lib} PROPERTIES LINKER_LANGUAGE CXX)
  endif()

  add_executable(${_chat} chat.c)
  target_link_libraries(${_chat} PRIVATE ${_lib})
  set_target_properties(${_chat} PROPERTIES OUTPUT_NAME ${_chat_out})

  add_executable(${_gen} generate_main.c)
  target_link_libraries(${_gen} PRIVATE ${_lib})
  set_target_properties(${_gen} PROPERTIES OUTPUT_NAME ${_gen_out})

  set(_exes ${_lib} ${_chat} ${_gen})

  if(NOT CMAKE_CROSSCOMPILING)
    if(SUFFIX)
      set(_tq test_quant_${BE})
      set(_tt test_tokenizer_${BE})
      set(_tp test_prompt_${BE})
      set(_tl test_linear_${BE})
      set(_tq_out test_quant-${BE})
      set(_tt_out test_tokenizer-${BE})
      set(_tp_out test_prompt-${BE})
      set(_tl_out test_linear-${BE})
    else()
      set(_tq test_quant)
      set(_tt test_tokenizer)
      set(_tp test_prompt)
      set(_tl test_linear)
      set(_tq_out test_quant)
      set(_tt_out test_tokenizer)
      set(_tp_out test_prompt)
      set(_tl_out test_linear)
    endif()

    add_executable(${_tq} tests/test_quant.c)
    target_link_libraries(${_tq} PRIVATE ${_lib})
    set_target_properties(${_tq} PROPERTIES OUTPUT_NAME ${_tq_out})
    add_test(NAME quant-${BE} COMMAND ${_tq})

    add_executable(${_tt} tests/test_tokenizer.c)
    target_link_libraries(${_tt} PRIVATE ${_lib})
    set_target_properties(${_tt} PROPERTIES OUTPUT_NAME ${_tt_out})
    add_test(NAME tokenizer-${BE} COMMAND ${_tt})

    add_executable(${_tp} tests/test_prompt.c)
    target_link_libraries(${_tp} PRIVATE ${_lib})
    set_target_properties(${_tp} PROPERTIES OUTPUT_NAME ${_tp_out})
    target_compile_definitions(${_tp} PRIVATE
      LLM_GOLDEN_DIR="${LLM_C_DIR}/tests/golden")
    add_test(NAME prompt-${BE} COMMAND ${_tp})

    add_executable(${_tl} tests/test_linear.c)
    target_link_libraries(${_tl} PRIVATE ${_lib})
    set_target_properties(${_tl} PROPERTIES OUTPUT_NAME ${_tl_out})
    target_compile_definitions(${_tl} PRIVATE
      LLM_GOLDEN_DIR="${LLM_C_DIR}/tests/golden")
    add_test(NAME linear-${BE} COMMAND ${_tl})

    list(APPEND _exes ${_tq} ${_tt} ${_tp} ${_tl})

    if(BE STREQUAL "pim")
      if(SUFFIX)
        set(_tplan test_planner_${BE})
        set(_tplan_out test_planner-${BE})
      else()
        set(_tplan test_planner)
        set(_tplan_out test_planner)
      endif()
      add_executable(${_tplan} "${LLM_C_DIR}/backends/pim/test_planner.cpp")
      target_link_libraries(${_tplan} PRIVATE ${_lib})
      target_include_directories(${_tplan} PRIVATE "${LLM_C_DIR}/backends/pim")
      set_target_properties(${_tplan} PROPERTIES OUTPUT_NAME ${_tplan_out})
      add_test(NAME planner-${BE} COMMAND ${_tplan})
      list(APPEND _exes ${_tplan})

      if(SUFFIX)
        set(_hstub pim_hybrid_stub_${BE})
        set(_hstub_out pim_hybrid_stub-${BE})
      else()
        set(_hstub pim_hybrid_stub)
        set(_hstub_out pim_hybrid_stub)
      endif()
      add_executable(${_hstub} "${LLM_C_DIR}/backends/pim/hybrid_stub.c")
      target_link_libraries(${_hstub} PRIVATE ${_lib})
      set_target_properties(${_hstub} PROPERTIES OUTPUT_NAME ${_hstub_out} LINKER_LANGUAGE CXX)
      list(APPEND _exes ${_hstub})
    endif()
  else()
    if(SUFFIX)
      set(_tl test_linear_${BE})
      set(_tl_out test_linear-${BE})
    else()
      set(_tl test_linear)
      set(_tl_out test_linear)
    endif()
    add_executable(${_tl} tests/test_linear.c)
    target_link_libraries(${_tl} PRIVATE ${_lib})
    set_target_properties(${_tl} PROPERTIES OUTPUT_NAME ${_tl_out})
    target_compile_definitions(${_tl} PRIVATE
      LLM_GOLDEN_DIR="${LLM_C_DIR}/tests/golden")
    list(APPEND _exes ${_tl})
    if(BE STREQUAL "pim")
      add_executable(pim_hybrid_stub "${LLM_C_DIR}/backends/pim/hybrid_stub.c")
      target_link_libraries(pim_hybrid_stub PRIVATE ${_lib})
      set_target_properties(pim_hybrid_stub PROPERTIES OUTPUT_NAME pim_hybrid_stub
                            LINKER_LANGUAGE CXX)
      list(APPEND _exes pim_hybrid_stub)
    endif()
  endif()

  if(BE STREQUAL "pim")
    foreach(_t ${_exes})
      set_target_properties(${_t} PROPERTIES LINKER_LANGUAGE CXX)
    endforeach()
  endif()
endfunction()

llm_available_backends(_LLM_HOST_BACKENDS)

if(LLM_BACKEND STREQUAL "all")
  set(_LLM_BUILD_BACKENDS ${_LLM_HOST_BACKENDS})
  set(_LLM_SUFFIX TRUE)
else()
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/backends/${LLM_BACKEND}/CMakeLists.txt")
    message(FATAL_ERROR
            "Unknown LLM_BACKEND=${LLM_BACKEND}. Use all, cpu, x86_64, aarch64, gpu, or pim")
  endif()
  list(FIND _LLM_HOST_BACKENDS "${LLM_BACKEND}" _be_idx)
  if(_be_idx LESS 0)
    message(FATAL_ERROR
            "LLM_BACKEND=${LLM_BACKEND} is not available on this host "
            "(have: ${_LLM_HOST_BACKENDS})")
  endif()
  set(_LLM_BUILD_BACKENDS ${LLM_BACKEND})
  set(_LLM_SUFFIX FALSE)
endif()

if("pim" IN_LIST _LLM_BUILD_BACKENDS)
  enable_language(CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT CMAKE_CROSSCOMPILING)
  enable_testing()
endif()

foreach(_be ${_LLM_BUILD_BACKENDS})
  llm_add_backend(${_be} ${_LLM_SUFFIX})
endforeach()

list(LENGTH _LLM_BUILD_BACKENDS _nbe)
message(STATUS "llm_infer backends: ${_LLM_BUILD_BACKENDS}")
if(_LLM_SUFFIX)
  message(STATUS "binaries: chat-<backend>, generate-<backend> (and tests)")
endif()
