// Copyright 2021 The KLW Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KLW_COMPILER_DIALECT_STREAM_IR_STREAMDIALECT_H_
#define KLW_COMPILER_DIALECT_STREAM_IR_STREAMDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace klw_compiler {
namespace KLW {
namespace Schedule {

class ScheduleDialect : public Dialect {
 public:
  explicit ScheduleDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "sch"; }

  void getCanonicalizationPatterns(RewritePatternSet &results) const override;

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
  void printAttribute(Attribute attr, DialectAsmPrinter &p) const override;

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &p) const override;

 private:
  void registerAttributes();
  void registerTypes();
};

}  // namespace Schedule
}  // namespace KLW
}  // namespace klw_compiler
}  // namespace mlir

#endif  // KLW_COMPILER_DIALECT_STREAM_IR_STREAMDIALECT_H_
