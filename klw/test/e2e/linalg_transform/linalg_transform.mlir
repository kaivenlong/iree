// RUN: iree-run-mlir --iree-hal-target-backends=llvm-cpu \
/// Specify the dispatch region formation with the transform dialect.
// RUN:   --iree-flow-dispatch-use-transform-dialect=%p/transform_dialect_dispatch_spec.mlir \
/// Specify the codegen strategy with the transform dialect.
// RUN:   --iree-codegen-llvmcpu-use-transform-dialect=%p/transform_dialect_codegen_spec.mlir \
// RUN: %s | FileCheck %s

#config_mnk_2304_8064_512 = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[128,128,16]]>,
  translation_info = <LLVMGPUMatmulTensorCore pipeline_depth=3>,
  workgroup_size = [128, 1, 1]>

func.func @matmul_static() -> tensor<2304x8064xf32> {
  %res = flow.tensor.constant dense<0.0> : tensor<2304x8064xf32> -> tensor<2304x8064xf32>
  %lhs = flow.tensor.constant dense<1.0> : tensor<2304x512xf32> -> tensor<2304x512xf32>
  %rhs = flow.tensor.constant dense<1.0> : tensor<512x8064xf32> -> tensor<512x8064xf32>

  %matmul = linalg.matmul {compilation_info = #config_mnk_2304_8064_512,
      lowering_config = #iree_codegen.lowering_config<tile_sizes=[[128,128,16]]>}
      ins(%lhs, %rhs : tensor<2304x512xf32>, tensor<512x8064xf32>)
      outs(%res : tensor<2304x8064xf32>) -> tensor<2304x8064xf32>
  %matmul_res = util.do_not_optimize(%matmul) : tensor<2304x8064xf32>

  return %matmul_res : tensor<2304x8064xf32>
}

//      CHECK: 2304x8064xf32=
