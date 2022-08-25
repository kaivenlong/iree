// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KLW_COMPILER_TOOLS_KLW_COMPILE_LIB_H
#define KLW_COMPILER_TOOLS_KLW_COMPILE_LIB_H

namespace mlir {
namespace klw_compiler {

int runKlwcMain(int argc, char **argv);

}  // namespace klw_compiler
}  // namespace mlir

#endif  // KLW_COMPILER_TOOLS_KLW_COMPILE_LIB_H
