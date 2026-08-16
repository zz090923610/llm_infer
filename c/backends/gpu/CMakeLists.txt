# Find Vulkan. Android NDK provides libvulkan; Linux host needs libvulkan1.
if(ANDROID)
  find_library(VULKAN_LIB vulkan)
  if(NOT VULKAN_LIB)
    message(FATAL_ERROR "NDK libvulkan not found (need ANDROID_ABI + platform sysroot)")
  endif()
else()
  find_library(VULKAN_LIB vulkan)
  if(NOT VULKAN_LIB)
    message(FATAL_ERROR
            "LLM_BACKEND=gpu requires Vulkan. Cross-compile with the Android NDK "
            "or install libvulkan-dev and retry.")
  endif()
endif()

set(LLM_BACKEND_LIBS ${VULKAN_LIB} PARENT_SCOPE)
set(LLM_BACKEND_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

set(LLM_BACKEND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/backend.c
    ${CMAKE_CURRENT_SOURCE_DIR}/vk.c
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor.c
    ${CMAKE_CURRENT_SOURCE_DIR}/rope.c
    ${CMAKE_CURRENT_SOURCE_DIR}/attn.c
    PARENT_SCOPE
)
