cmake_minimum_required(VERSION 3.20)
project(devsim_openblas LANGUAGES C)

add_library(devsim_openblas_shim SHARED src/shim.c)
set_target_properties(devsim_openblas_shim PROPERTIES C_VISIBILITY_PRESET hidden)
if(UNIX AND NOT APPLE)
  target_link_libraries(devsim_openblas_shim PRIVATE ${CMAKE_DL_LIBS})
endif()

install(TARGETS devsim_openblas_shim
  LIBRARY DESTINATION devsim_openblas
  RUNTIME DESTINATION devsim_openblas)
