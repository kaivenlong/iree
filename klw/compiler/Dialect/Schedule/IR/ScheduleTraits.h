// Copyright 2021 The KLW Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KLW_COMPILER_DIALECT_SCHEDULE_IR_SCHEDULETRAITS_H_
#define KLW_COMPILER_DIALECT_SCHEDULE_IR_SCHEDULETRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace KLW {
namespace Schedule {

template <typename ConcreteType>
class TensorPhaseOp : public OpTrait::TraitBase<ConcreteType, TensorPhaseOp> {
 public:
  static LogicalResult verifyTrait(Operation *op) { return success(); }
};

template <typename ConcreteType>
class AsyncPhaseOp : public OpTrait::TraitBase<ConcreteType, AsyncPhaseOp> {
 public:
  static LogicalResult verifyTrait(Operation *op) { return success(); }
};

template <typename ConcreteType>
class CmdPhaseOp : public OpTrait::TraitBase<ConcreteType, CmdPhaseOp> {
 public:
  static LogicalResult verifyTrait(Operation *op) { return success(); }
};

}  // namespace Schedule
}  // namespace KLW
}  // namespace OpTrait
}  // namespace mlir

#endif  // KLW_COMPILER_DIALECT_SCHEDULE_IR_SCHEDULETRAITS_H_
