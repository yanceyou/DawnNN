# ---------------------------------------------------------------------------------------------------------------------
# Build SHARED/STATIC libs
# ---------------------------------------------------------------------------------------------------------------------
add_library(dawnnn STATIC)
if (DAWNNN_SHARED_LIB)
  add_library(dawnnn SHARED)
endif()

target_sources(dawnnn
  PUBLIC
    "Layer.h"
  PRIVATE
    "Layer.cpp"
)

message("-- dawnnn_SOURCE_DIR: ${dawnnn_SOURCE_DIR}/include")
target_include_directories(dawnnn
  PUBLIC
    "${dawnnn_SOURCE_DIR}/include"
)

target_link_libraries(dawnnn PRIVATE
  dawn_internal_config
  dawncpp
  dawn_proc
  dawn_common
  dawn_glfw
  dawn_native
  dawn_wire
  dawn_utils
)
