// Copyright 2022 The KLW Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KLW_COMPILER_CODEGEN_KLWGPU_TRANSFORMEXTENSIONS_KLWGPUEXTENSIONS_H_
#define KLW_COMPILER_CODEGEN_KLWGPU_TRANSFORMEXTENSIONS_KLWGPUEXTENSIONS_H_

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

namespace mlir {
class DialectRegistry;

namespace klw_compiler {

/// Registers Flow transformations that require KLW-specific information into
/// the transform dialect.
void registerTransformDialectKLWGPUExtension(DialectRegistry &registry);

namespace KLW {
namespace transform_dialect {
// Hook to register KLWGPU transformations to the transform dialect.
class KLWGPUExtensions
    : public transform::TransformDialectExtension<KLWGPUExtensions> {
 public:
  KLWGPUExtensions();
};

/// Outline linalg ops
void populateLinalgToFunctionCallConversionPatterns(RewritePatternSet &patterns);


}  // namespace transform_dialect
}  // namespace KLW
}  // namespace klw_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "klw/compiler/Dialect/Schedule/TransformExtensions/KLWGPUExtensionsOps.h.inc"

#endif  // KLW_COMPILER_CODEGEN_KLWGPU_TRANSFORMEXTENSIONS_KLWGPUEXTENSIONS_H_
