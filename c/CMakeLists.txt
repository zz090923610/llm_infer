cmake_minimum_required(VERSION 3.16)
project(llm_infer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)
add_compile_definitions(_POSIX_C_SOURCE=200809L)

get_filename_component(LLM_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

set(LLM_SOURCES
    util.c
    hashmap.c
    heap.c
    tensor.c
    quant.c
    gguf.c
    unicode.c
    unicode_data.c
    tokenizer.c
    rope.c
    cache.c
    model.c
    sampler.c
    generate.c
)

add_library(llm STATIC ${LLM_SOURCES})
target_include_directories(llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llm PUBLIC m)
target_compile_definitions(llm PUBLIC LLM_DEFAULT_MODEL="${LLM_REPO_ROOT}/nanogpt-chat-q8_0.gguf")

add_executable(chat chat.c)
target_link_libraries(chat PRIVATE llm)

add_executable(generate generate_main.c)
target_link_libraries(generate PRIVATE llm)

enable_testing()

add_executable(test_quant test_quant.c)
target_link_libraries(test_quant PRIVATE llm)
add_test(NAME quant COMMAND test_quant)

add_executable(test_tokenizer test_tokenizer.c)
target_link_libraries(test_tokenizer PRIVATE llm)
add_test(NAME tokenizer COMMAND test_tokenizer)
