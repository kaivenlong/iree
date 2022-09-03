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

#define DEBUG_TYPE "klw-outline-linalg-op"

#define dbgs_v() (llvm::dbgs() << "[" << __FUNCTION__ << " " << __LINE__ << "]")

namespace mlir {
namespace klw_compiler {
namespace KLW {
namespace transform_dialect {

static FlatSymbolRefAttr getCustomFuncSymbolRef(Operation *op,
  PatternRewriter &rewriter) {
  auto linalgOp = cast<LinalgOp>(op);
  auto fnName = linalg.get
}

class LinalgOpToFunctionCallRewrite
  : public OpInterfaceRewritePattern<LinalgOp> {
public:
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(LinalgOp op,
    PatternRewriter &rewriter) const override {
    if (!isa<linalg::MatmulOp, linalg::Conv2DNhwcHwcfOp>(op)) {
      dbgs_v() << "warning: non-supported op: \n" << op << "\n";
      return failure();
    }
    auto customFuncName = getCustomFuncSymbolRef(op, rewriter);
    if (!customFuncName) {
      dbgs_v() << "warning: failed to get function name for :\n"
        << op << "\n\n";
      return failure();
    }
    dbgs_v() << "about to replace : \n" << op
      << "\n" << " with call func \n" << customFuncName.getValue() << "\n\n";
    auto compilationInfo = mlir::iree_compiler::getCompilationInfo(op.getOperation());
    auto loweringConfig = mlir::iree_compiler::getLoweringConfig(op.getOperation());
    auto newOp = rewriter.replaceOpWithNewOp<func::CallOp>(op, customFuncName.getValue(),
      TypeRange(), createTypeCanonicalizedMemRefOperands(rewriter, op->getLoc(),
      op->getOperands()));
    if (loweringConfig) mlir::iree_compiler::setLoweringConfig(newOp, loweringConfig);
    if (compilationInfo) mlir::iree_compiler::setCompliationInfo(newOp, compilationInfo);
    setCustomScheduleFuncName(newOp, customFuncName.getValue());
    if (auto convOp = dyn_cast<linalg::Conv2DNhwcHwcfOp>(*op.getOperation())) {
      auto strides = convOp.stridesAttr();
      auto dilations = convOp.dilationsAttr();
      newOp->setAttr(getDilationsAttrName(), dilations);
      newOp->setAttr(getStridesAttrName(), strides);
    }
    dbgs_v() << "already replace: \n" << op << "\n" << " with call func \n"
      << customFuncName.getValue() << "\n\n";
    dbgs_v() << "newOp = \n" << newOp << "\n\n";
    return success();
  }
};

void populateLinalgToFunctionCallConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<LinalgOpToFunctionCallRewrite>(patterns.getContext());
}

}  // namespace transform_dialect
}  // namespace KLW
}  // namespace klw_compiler
}  // namespace mlir

