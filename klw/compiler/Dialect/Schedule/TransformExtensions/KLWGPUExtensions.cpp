// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "KLWGPUExtensions.h"
#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "klw-transform-extensions"

#define dbgs_v() (llvm::dbgs() << "[" << __FUNCTION__ << " " << __LINE__ << "]")

using namespace mlir;
//using namespace mlir::iree_compiler::IREE;
using namespace mlir::klw_compiler::KLW;

klw_compiler::KLW::transform_dialect::KLWGPUExtensions::KLWGPUExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "klw/compiler/Dialect/Schedule/TransformExtensions/KLWGPUExtensionsOps.cpp.inc"
      >();
}

void mlir::klw_compiler::registerTransformDialectKLWGPUExtension(
    DialectRegistry &registry) {
  registry.addExtensions<transform_dialect::KLWGPUExtensions>();
}

//===---------------------------------------------------------------------===//
// KLW-specific transformations.
//===---------------------------------------------------------------------===//
namespace mlir {
namespace iree_compiler {
extern llvm::cl::opt<std::string> clGPUCodegenTransformDialectFileName;
}
}

DiagnosedSilenceableFailure
transform_dialect::ReplaceOpWithCustomScheduleOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  dbgs_v() << "ReplaceOpWithCustomScheduleOp: target is \n" << *target << "\n";
  return DiagnosedSilenceableFailure(success());
}

#define GET_OP_CLASSES
#include "klw/compiler/Dialect/Schedule/TransformExtensions/KLWGPUExtensionsOps.cpp.inc"
