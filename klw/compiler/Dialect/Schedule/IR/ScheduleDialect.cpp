// Copyright 2021 The KLW Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "klw/compiler/Dialect/Schedule/IR/ScheduleDialect.h"
#include "klw/compiler/Dialect/Schedule/IR/ScheduleOps.h"
#include "klw/compiler/Dialect/Schedule/IR/ScheduleTypes.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace klw_compiler {
namespace KLW {
namespace Schedule {

namespace {

// Used to control inlining behavior.
struct ScheduleInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Sure!
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    // Sure!
    return true;
  }
};

struct ScheduleFolderInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  bool shouldMaterializeInto(Region *region) const override {
    // TODO(benvanik): redirect constants to the region scope when small.
    return false;
  }
};

// Tries to fold away unrealized_conversion_cast ops if the downstream consumers
// don't need the extra information. These are inserted during conversion or
// transforms that may interop with external dialects.
//
// Specifically matches:
//   %0 = builtin.unrealized_conversion_cast %arg0, %arg1 :
//        !stream.resource<transient>, index to !stream.resource<transient>
struct StripResourceConversionCastPattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(UnrealizedConversionCastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto result = castOp.getResult(0);
    if (!result.getType().isa<KLW::Schedule::ResourceType>()) return failure();
    assert(castOp.getNumOperands() == 2 &&
           "expect resource, index -> resource");
    auto resourceValue = castOp.getOperand(0);
    auto sizeValue = castOp.getOperand(1);
    for (auto &use : llvm::make_early_inc_range(result.getUses())) {
      if (auto sizeOp =
              dyn_cast<KLW::Schedule::ResourceSizeOp>(use.getOwner())) {
        sizeOp.getResult().replaceAllUsesWith(sizeValue);
        rewriter.eraseOp(sizeOp);
      } else {
        use.set(resourceValue);
      }
    }
    rewriter.eraseOp(castOp);
    return success();
  }
};

}  // namespace

ScheduleDialect::ScheduleDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<ScheduleDialect>()) {
  registerAttributes();
  registerTypes();

#define GET_OP_LIST
  addOperations<
#include "klw/compiler/Dialect/Schedule/IR/ScheduleOps.cpp.inc"
      >();
  addInterfaces<ScheduleInlinerInterface, ScheduleFolderInterface>();
}

void ScheduleDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.insert<StripResourceConversionCastPattern>(getContext());
}

Operation *ScheduleDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  if (mlir::func::ConstantOp::isBuildableWith(value, type)) {
    return builder.create<mlir::func::ConstantOp>(
        loc, type, value.cast<FlatSymbolRefAttr>());
  } else if (arith::ConstantOp::isBuildableWith(value, type)) {
    return builder.create<arith::ConstantOp>(loc, type, value);
  } else if (value.isa<KLW::Schedule::TimepointAttr>()) {
    return builder.create<KLW::Schedule::TimepointImmediateOp>(loc);
  }
  return nullptr;
}

}  // namespace Schedule
}  // namespace KLW
}  // namespace klw_compiler
}  // namespace mlir
