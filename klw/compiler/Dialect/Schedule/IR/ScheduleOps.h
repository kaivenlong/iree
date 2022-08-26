// Copyright 2021 The KLW Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KLW_COMPILER_DIALECT_SCHEDULE_IR_SCHEDULEOPS_H_
#define KLW_COMPILER_DIALECT_SCHEDULE_IR_SCHEDULEOPS_H_

#include <cstdint>

#include "klw/compiler/Dialect/Schedule/IR/ScheduleDialect.h"
#include "klw/compiler/Dialect/Schedule/IR/ScheduleTraits.h"
#include "klw/compiler/Dialect/Schedule/IR/ScheduleTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;
using namespace mlir::iree_compiler;

#define GET_OP_CLASSES
#include "klw/compiler/Dialect/Schedule/IR/ScheduleOps.h.inc"  // IWYU pragma: export

#endif  // KLW_COMPILER_DIALECT_SCHEDULE_IR_SCHEDULEOPS_H_
