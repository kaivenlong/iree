builddir=iree_build
cmake --build $builddir --target iree-test-deps
ctest --test-dir $builddir --output-on-failure

