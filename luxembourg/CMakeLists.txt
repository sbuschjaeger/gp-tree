cmake_minimum_required(VERSION 3.8)
project(LuxembourgExperiment) #TODO RENAME

set(CMAKE_CXX_STANDARD 17)
# At the momement we dont care about unused variables
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-unused-variable -g")

# -funroll-loops -ftree-vectorize -O3  -pthread
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")   # O3 is default
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ")

# We want to profile atm..
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

# Specify options
option(USE_NATIVE "Compile with [march|mtune]=native.   " OFF)
option(USE_OPENMP "Build with OpenMP Support.           " OFF)
option(USE_MKL "Build with Intel MKL Support.           " OFF)
option(USE_TORCH "Build with Torch NN Support.          " OFF)

#message(" --CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")

# For docker
if (USE_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
endif ()

set(INCLUDE_DIR "../submodules/ensembles/code/include")

set(SRC_H 
    ${INCLUDE_DIR}/DecisionStump.h
    ${INCLUDE_DIR}/Model.h
    ${INCLUDE_DIR}/BatchLearner.h
    ${INCLUDE_DIR}/Dataset.h
    ${INCLUDE_DIR}/Utils.h
    ${INCLUDE_DIR}/Backend.h
    ${INCLUDE_DIR}/Splitter.h
    ${INCLUDE_DIR}/DefaultModel.h
    ${INCLUDE_DIR}/ModelTree.h
    ${INCLUDE_DIR}/DecisionTree.h
    ${INCLUDE_DIR}/GaussianProcess.h
    ${INCLUDE_DIR}/InformativeVectorMachine.h
	${INCLUDE_DIR}/GaussianModelTree.h
)

if (USE_OPENMP)
    message("-- Enabling OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    add_definitions(-DUSE_OPENMP)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        message("-- Found GCC compiler. Enabling _GLIBCXX_PARALLEL")
        add_definitions(-D_GLIBCXX_PARALLEL)
    endif ()
else ()
    # -Wall activates -Wunknown-pragmas, which leads to a few unknown openmp pragmas
    # since we just disabled openmp. Thus we disable these warning here
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif ()

if (USE_MKL AND NOT CROSS_COMPILE_TARGET_ARCH)
    message("-- Using Intel MKL Support.")
    # Note: We use MY_MKL_ROOT to not interfere with the "real" MKL_ROOT from CMakes find_package
    set(MY_MKL_ROOT "/opt/intel/compilers_and_libraries/linux/mkl")
    if (EXISTS ${MY_MKL_ROOT})
        message("-- MKL_ROOT exists. (${MY_MKL_ROOT})")
        add_definitions(-DUSE_MKL)

        include_directories("${MY_MKL_ROOT}/include")
        # Note: Changed lmkl_intel_ilp64 -> lmkl_intel_lp64. If we want to compile pytorch with ilp, we could try adding -DMKL_ILP64 to cmake
        # Note: In order to make the python bindings work with MKL, I added lmkl_avx2.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${MY_MKL_ROOT}/lib/intel64 -Wl,--no-as-needed  -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -lmkl_avx2")
    else ()
        message("-- [ERROR]: MKL_ROOT not found. (${MY_MKL_ROOT})")
    endif ()
endif ()

if (USE_TORCH AND NOT CROSS_COMPILE_TARGET_ARCH)
	set(CMAKE_PREFIX_PATH "/opt/libtorch/intel-nightly")
	message("-- Using Torch support.")
	find_package(Torch REQUIRED)

	set(SRC_H ${SRC_H}
		${INCLUDE_DIR}/TorchWrapper.h
	)

    add_definitions(-DUSE_TORCH)
endif ()

add_executable(main
        main.cpp ${SRC_CC} ${SRC_C} ${SRC_H}
)
target_include_directories(main PRIVATE ${INCLUDE_DIR})

if (USE_TORCH)
	target_link_libraries(main "${TORCH_LIBRARIES}")
endif ()
