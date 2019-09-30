
cmake_minimum_required (VERSION 3.9)
project (BudgeteddualSVM_SW)

if (APPLE)
	SET(MACOSX_DEPLOYMENT_TARGET ${DARWIN_MAJOR_SDK_VERSION}.${DARWIN_MINOR_SDK_VERSION})
	SET(CMAKE_CXX_FLAGS "--sysroot ${CMAKE_OSX_SYSROOT} ${CMAKE_CXX_FLAGS}")
	MESSAGE("Setting MACOSX_DEPLOYMENT_TARGET to '${MACOSX_DEPLOYMENT_TARGET}'.")
endif (APPLE)

# directories
set (BSVMSrcDir "${PROJECT_SOURCE_DIR}/src")
#include_directories(BSVMSrcDir)
include_directories(${BSVMSrcDir})

# we use c++11
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	add_definitions ( "-DDEBUG" )
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -g")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -g")
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
	add_definitions ( "-DDEBUG" )
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release" )
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")
endif()
message( STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}" )

# executable	
set (COMMON_HEADER
	${BSVMSrcDir}/solver.h
	${BSVMSrcDir}/loadData.h
	${BSVMSrcDir}/kernel.h
	${BSVMSrcDir}/budgetMaintenance.h
	${BSVMSrcDir}/svm.h
)

set (COMMON_FILES
	${BSVMSrcDir}/loadData.cpp
	${BSVMSrcDir}/budgetMaintenance.cpp
	${BSVMSrcDir}/solver.cpp
)

# link things
add_library (filesDirectory ${COMMON_FILES} ${COMMON_HEADER})
# target_link_libraries(BSVM filesDirectory)
add_executable (BSVMO ${BSVMSrcDir}/main.cpp)
target_link_libraries (BSVMO filesDirectory ${SHARK_LIBRARIES})

# install things	
# install (TARGETS BSVMO DESTINATION bin)
include (InstallRequiredSystemLibraries)
set (CPACK_PACKAGE_VERSION_MAJOR "${SP_SVM_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${SP_SVM_VERSION_MINOR}")
include (CPack)

