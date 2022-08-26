add_subdirectory(klw/compiler)
add_subdirectory(klw/tools)
include_directories(klw)
if (${IREE_BUILD_KLW})
  add_definitions(-DIREE_BUILD_KLW)
endif()

