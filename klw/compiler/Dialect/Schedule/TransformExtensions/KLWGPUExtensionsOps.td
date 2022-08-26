// Copyright 2022 The KLW Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KLW_COMPILER_CODEGEN_LLVMGPU_TRANSFORMEXTENSIONS_LLVMGPUEXTENSIONS
#define KLW_COMPILER_CODEGEN_LLVMGPU_TRANSFORMEXTENSIONS_LLVMGPUEXTENSIONS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ReplaceOpWithCustomScheduleOp :
  Op<Transform_Dialect, "schedule.replace_op_with_custom_schedule",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformEachOpTrait, 
     TransformOpInterface]> {
  let description = [{
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<StrAttr, "">:$schedule);
  let results = (outs);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::klw_compiler::KLW::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}
#endif // KLW_COMPILER_CODEGEN_LLVMGPU_TRANSFORMEXTENSIONS_LLVMGPUEXTENSIONS
