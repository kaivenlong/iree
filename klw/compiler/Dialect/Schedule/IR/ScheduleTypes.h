// Copyright 2021 The KLW Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KLW_COMPILER_DIALECT_SCHEDULE_IR_SCHEDULETYPES_H_
#define KLW_COMPILER_DIALECT_SCHEDULE_IR_SCHEDULETYPES_H_

#include "klw/compiler/Dialect/Schedule/IR/ScheduleDialect.h"
#include "klw/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#include "klw/compiler/Dialect/Schedule/IR/ScheduleEnums.h.inc"  // IWYU pragma: export
// clang-format on

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "klw/compiler/Dialect/Schedule/IR/ScheduleAttrs.h.inc"  // IWYU pragma: keep
// clang-format on

namespace mlir {
namespace klw_compiler {
namespace KLW {
namespace Schedule {

class AffinityAttr;

#include "klw/compiler/Dialect/Schedule/IR/ScheduleTypeInterfaces.h.inc"  // IWYU pragma: export

}  // namespace Schedule
}  // namespace KLW
}  // namespace klw_compiler
}  // namespace mlir

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_TYPEDEF_CLASSES
#include "klw/compiler/Dialect/Schedule/IR/ScheduleTypes.h.inc"  // IWYU pragma: keep
// clang-format on

namespace mlir {
namespace klw_compiler {
namespace KLW {
namespace Schedule {

#include "klw/compiler/Dialect/Schedule/IR/ScheduleOpInterfaces.h.inc"  // IWYU pragma: export

}  // namespace Schedule
}  // namespace KLW
}  // namespace klw_compiler
}  // namespace mlir

#endif  // KLW_COMPILER_DIALECT_SCHEDULE_IR_SCHEDULETYPES_H_
