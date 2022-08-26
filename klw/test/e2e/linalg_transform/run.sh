bindir=../../../../iree_build/klw/tools
workdir=work
mkdir -p $workdir
build(){
  tag=$workdir/$1
  $bindir/klw-compile \
    --mlir-disable-threading \
    --mlir-print-ir-before-all \
    --mlir-print-ir-after-all \
    --iree-hal-target-backends=cuda \
    --iree-hal-cuda-llvm-target-arch=sm_80 \
    --iree-codegen-llvmgpu-use-transform-dialect=$2 \
    $1 -o $tag.cuda.vmfb &> $tag.build.log
}

build linalg_transform.mlir codegen_spec.mlir
 
