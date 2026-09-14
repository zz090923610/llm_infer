# Host-side PIM GEMV via pim_func. Decode n_tok==1 linear stays command-stepped.
# On x86_64, attn/RoPE/prefill reuse the AVX2 kernels + pthread pool.

set(_PIM_FUNC_DIR "")
foreach(_cand
        "${LLM_C_DIR}/../../pim_func"
        "${LLM_C_DIR}/../pim_func"
        "${LLM_C_DIR}/../../gem5_se/pim_func")
  get_filename_component(_abs "${_cand}" ABSOLUTE)
  if(EXISTS "${_abs}/CMakeLists.txt")
    set(_PIM_FUNC_DIR "${_abs}")
    break()
  endif()
endforeach()
if(NOT _PIM_FUNC_DIR)
  message(FATAL_ERROR
          "LLM_BACKEND=pim needs pim_func. Tried next to gem5_se or under gem5_se/")
endif()

if(NOT TARGET pim_func)
  add_subdirectory("${_PIM_FUNC_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/pim_func")
endif()

set(LLM_BACKEND_LIBS pim_func PARENT_SCOPE)
set(LLM_BACKEND_INCLUDE "${_PIM_FUNC_DIR}/include" PARENT_SCOPE)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(LLM_BACKEND_COMPILE_OPTIONS -mavx2 -mfma PARENT_SCOPE)
  set(LLM_BACKEND_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/backend.c
      ${CMAKE_CURRENT_SOURCE_DIR}/tensor.c
      ${CMAKE_CURRENT_SOURCE_DIR}/cpu_kernels.c
      ${CMAKE_CURRENT_SOURCE_DIR}/planner.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/gemv.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../host/pc/x86_64-simd/pool.c
      ${CMAKE_CURRENT_SOURCE_DIR}/../host/pc/x86_64-simd/rope.c
      ${CMAKE_CURRENT_SOURCE_DIR}/../host/pc/x86_64-simd/attn.c
      PARENT_SCOPE
  )
else()
  set(LLM_BACKEND_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/backend.c
      ${CMAKE_CURRENT_SOURCE_DIR}/tensor.c
      ${CMAKE_CURRENT_SOURCE_DIR}/cpu_kernels.c
      ${CMAKE_CURRENT_SOURCE_DIR}/planner.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/gemv.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../host/pc/plain-cpu/rope.c
      ${CMAKE_CURRENT_SOURCE_DIR}/../host/pc/plain-cpu/attn.c
      PARENT_SCOPE
  )
endif()
