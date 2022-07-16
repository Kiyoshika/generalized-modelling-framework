cmake_minimum_required(VERSION 3.8)

project(GMF C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "-O0 -g -Wall")

# CMatrix dependency
add_subdirectory(ext/CMatrix)

# toggle whether or not to compile examples
option(EXAMPLES OFF)

add_subdirectory(src)
