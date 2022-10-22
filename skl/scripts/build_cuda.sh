# Recommended for simple development using clang and lld:
builddir=iree_build
mkdir -p $builddir
cmake -GNinja -B $builddir -S . \
    -DCMAKE_BUILD_TYPE=Debug \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang-12 \
    -DIREE_HAL_DRIVER_CUDA=ON \
    -DCMAKE_CXX_COMPILER=clang++-12 \
    -DIREE_ENABLE_LLD=ON

# Alternately, with system compiler and your choice of CMake generator:
# cmake -B ../iree-build/ -S .

# Additional quality of life CMake flags:
# Enable ccache:
#   -DIREE_ENABLE_CCACHE=ON

cmake --build $builddir -j32

