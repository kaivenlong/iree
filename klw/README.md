This folder contains the extension projects based on iree and the mlir core.
Follow iree's convention on coding and directory layout.

#Download source code

```shell
git clone --recurse-submodules git@github.com:kaivenlong/iree.git -b schedule_ir_dev sch_dev
```

#Build

```shell
# Recommended for simple development using clang and lld:
cmake -GNinja -B ../iree-build/ -S . \
    -DCMAKE_BUILD_TYPE=Debug \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON \
    -DIREE_HAL_DRIVER_CUDA_DEFAULT=ON \
    -DIREE_HAL_DRIVER_CUDA=ON \
    -DIREE_TARGET_BACKEND_CUDA=ON \
    -DIREE_BUILD_KLW=ON
```

#Test

Got iree_build
```shell
ctest --test-dir klw/compiler/Dialect/Schedule/IR/test
```

