bindir=../../../../iree_build/klw/tools

build(){
  $bindir/klw-compile \
    --mlir-disable-threading \
    --mlir-print-ir-before-all \
    --mlir-print-ir-after-all \
    --iree-hal-target-backends=cuda \
    --iree-hal-cuda-llvm-target-arch=sm_80 \
    --iree-codegen-llvmgpu-use-transform-dialect=$2 \
    $1 -o $1.cuda.vmfb &> $1.build.log
}

build linalg_transform.mlir transform_dialect_codegen_spec.mlir
 
