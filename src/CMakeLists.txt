# ---------------------------------------------------------------------------------------------------------------------
# Build SHARED/STATIC libs
# ---------------------------------------------------------------------------------------------------------------------
add_library(dawnnn STATIC)
if (DAWNNN_SHARED_LIB)
  add_library(dawnnn SHARED)
endif()

target_sources(dawnnn
  INTERFACE
  PRIVATE
    "Layer.cpp"
    "Layer.h"
)
