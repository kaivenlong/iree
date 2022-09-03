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

#define DEBUG_TYPE "klw-custom-schedule"

#define dbgs_v() (llvm::dbgs() << "[" << __FUNCTION__ << " " << __LINE__ << "]")

namespace mlir {
namespace klw_compiler {
namespace KLW {
namespace transform_dialect {

static FailureOr<func::FuncOp> createScheduleFunction(func::FuncOp &funcOp,
  Operation &targetOp, func::FuncOp &schTemplate) {
  func::FuncOp sch;
  MLIRContext *context = funcOp.getContext();
  ConversionTarget target(*context);
  RewritePatternSet patterns(context);
  target.addLegalDialect<AffineDialect, arith::ArithmeticDialect,
    func::FuncDialect, memref::MemRefDialect,
    mlir::iree_compiler::IREE::HAL::HALDialect, scf::SCFDialect,
    bufferization::BufferizationDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp,
    linalg::FillOp, linalg::GenericOp, linalg::YieldOp>();
  populateLinalgToFunctionCallConversionPatterns(patterns);
  if (failed(applyFullConversion(funcOp, target, std::move(patterns)))) {
    return failure();
  }
}

LogicalResult replaceOpWithCustomSchedule(Operation &target,
  func::FuncOp &schTemplate) {
  func::FuncOp funcOp = target.getParentOfType<func::FuncOp>();

  //  step 1. change linalg op into function call.
  auto sch = createScheduleFunction(funcOp, target, schTemplate);

  //  step 2. generate correct logic according to the schedule template
  //  function
  KLW::Schedule::createScheduleFunctionBody(funcOp, target,
    sch.getValue(), schTemplate);
  return success();
}

}  // namespace transform_dialect
}  // namespace KLW
}  // namespace klw_compiler
}  // namespace mlir

