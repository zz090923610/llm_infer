cmake_minimum_required(VERSION 3.16)
project(pim_func CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(pim_func STATIC
    src/isa.cpp
    src/layout.cpp
    src/mem.cpp
    src/gb.cpp
    src/pu.cpp
    src/device.cpp
    src/hook.cpp
    src/c_api.cpp
    src/mac_simd.cpp
    src/pool.cpp
    src/hybrid.cpp
)
target_include_directories(pim_func PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
option(PIM_DEFAULT_ISSUE_MMIO "Default PIM_ISSUE to mmio (gem5 guest). Off = host Device." OFF)
if(PIM_DEFAULT_ISSUE_MMIO)
  target_compile_definitions(pim_func PRIVATE PIM_DEFAULT_ISSUE_MMIO=1)
endif()
find_package(Threads REQUIRED)
target_link_libraries(pim_func PUBLIC Threads::Threads)
if(UNIX AND NOT APPLE)
  target_link_libraries(pim_func PUBLIC rt)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set_source_files_properties(src/mac_simd.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()

set(LLM_LINEAR_DUMP "${CMAKE_CURRENT_SOURCE_DIR}/../c/tests/golden/linear_16x64.txt")
if(NOT EXISTS "${LLM_LINEAR_DUMP}")
  set(LLM_LINEAR_DUMP "${CMAKE_CURRENT_SOURCE_DIR}/../llm_infer/c/tests/golden/linear_16x64.txt")
endif()
if(NOT EXISTS "${LLM_LINEAR_DUMP}")
  set(LLM_LINEAR_DUMP "${CMAKE_CURRENT_SOURCE_DIR}/../../llm_infer/c/tests/golden/linear_16x64.txt")
endif()

function(pim_func_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE pim_func)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_compile_definitions(${name} PRIVATE LLM_LINEAR_DUMP="${LLM_LINEAR_DUMP}")
  add_test(NAME ${name} COMMAND ${name})
endfunction()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  enable_testing()
  pim_func_test(test_isa)
  pim_func_test(test_layout)
  pim_func_test(test_gb)
  pim_func_test(test_parallelism)
  pim_func_test(test_linear_oracle)
  pim_func_test(test_hook)
  pim_func_test(test_c_api)
  pim_func_test(test_hybrid)
endif()
