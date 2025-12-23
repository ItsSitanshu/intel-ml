cmake_minimum_required(VERSION 3.23)
Project(intel-ml VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

add_executable(intel-ml)

target_sources(intel-ml 
  PRIVATE src/main.cxx
)

target_include_directories(intel-ml
  PRIVATE src/
)

set_target_properties(intel-ml PROPERTIES
  CXX_STANDARD 23
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)