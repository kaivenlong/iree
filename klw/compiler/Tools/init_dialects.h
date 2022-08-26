// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This files defines a helper to trigger the registration of dialects to
// the system.
//
// Based on MLIR's InitAllDialects but for IREE dialects.

#ifndef KLW_COMPILER_TOOLS_INIT_DIALECTS_H_
#define KLW_COMPILER_TOOLS_INIT_DIALECTS_H_

#include "iree/compiler/Tools/init_dialects.h"

namespace mlir {
namespace klw_compiler {

inline void registerAllDialects(DialectRegistry &registry) {
  mlir::iree_compiler::registerAllDialects(registry);
}

}  // namespace klw_compiler
}  // namespace mlir

#endif  // KLW_COMPILER_TOOLS_INIT_DIALECTS_H_
