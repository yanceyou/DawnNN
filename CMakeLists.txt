cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(dawnnn)

# ---------------------------------------------------------------------------------------------------------------------
# Set C/C++ compiler flags
# TODO: Use Modern C++ 20
#   - CMake Support: https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
#   - CXX modules: https://cmake.org/cmake/help/latest/manual/cmake-cxxmodules.7.html
# TODO: Check C++20 support compiler version
# ---------------------------------------------------------------------------------------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Modern Compiler: GCC|Clang|MSVC
# Relations between platforms and compilers
# Linux   | COMPILER_IS_GCC      | GNU
# Linux   | COMPILER_IS_CLANG    | Clang + libstdc++
# Linux   | COMPILER_IS_CLANG    | Clang + libc++
# Windows | COMPILER_IS_MSVC     | Visual Studio(MSVC)
# Windows | COMPILER_IS_CLANG_CL | Visual Studio With Clang(Clang-CL)
# MacOS   | COMPILER_IS_CLANG    | Apple Clang
# MacOS   | COMPILER_IS_CLANG    | Apple Clang(ARM64)
# Android | COMPILER_IS_GCC      | NDK With GNU
# Android | COMPILER_IS_CLANG    | NDK With Clang
# iOS     | COMPILER_IS_CLANG    | Apple Clang(ARM64)
if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  if (CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    set(COMPILER_IS_CLANG_CL TRUE)
  else()
    set(COMPILER_IS_CLANG TRUE)
  endif()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(COMPILER_IS_GCC TRUE)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(COMPILER_IS_CLANG TRUE)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(COMPILER_IS_MSVC TRUE)
endif()

# Compiler Support: https://en.cppreference.com/w/cpp/compiler_support/20
#   - GCC: https://gcc.gnu.org/projects/cxx-status.html
#   - CLANG: https://clang.llvm.org/cxx_status.html
#   - MSVC: https://devblogs.microsoft.com/cppblog/msvc-cpp20-and-the-std-cpp20-switch/
#   - APPLE CLANG : https://developer.apple.com/xcode/cpp/
message("-- CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

# ---------------------------------------------------------------------------------------------------------------------
# Build Options
# ---------------------------------------------------------------------------------------------------------------------
option(DAWNNN_SHARED_LIB "build shared library" OFF)
option(DAWNNN_BUILD_EXAMPLES "build examples" ON)

# ---------------------------------------------------------------------------------------------------------------------
# Build with all subdirectories
# ---------------------------------------------------------------------------------------------------------------------
add_subdirectory(third_party)
add_subdirectory(src)

# ---------------------------------------------------------------------------------------------------------------------
# Build executors
# ---------------------------------------------------------------------------------------------------------------------
if (DAWNNN_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
