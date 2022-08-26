// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry function for klw-opt and derived binaries.
//
// Based on mlir-opt but registers the passes and dialects we care about.

#include "klw/compiler/Tools/init_klw_dialects.h"
#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/Tools/init_passes.h"
#include "iree/compiler/Tools/init_targets.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  mlir::klw_compiler::registerKlwDialects(registry);
  mlir::iree_compiler::registerAllPasses();
  mlir::iree_compiler::registerHALTargetBackends();

  // Register the pass to drop embedded transform dialect IR.
  // TODO: this should be upstreamed.
  mlir::linalg::transform::registerDropSchedulePass();

  if (failed(MlirOptMain(argc, argv, "IREE-based KLW modular optimizer driver\n",
                         registry,
                         /*preloadDialectsInContext=*/false))) {
    return 1;
  }
  return 0;
}
