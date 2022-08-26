# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_tablegen.bzl", "iree_gentbl_cc_library", "iree_tablegen_doc")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "ScheduleBase.td",
            "ScheduleInterfaces.td",
            "ScheduleOps.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "//compiler/src/iree/compiler/Dialect/Util/IR:td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:SubElementInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

iree_compiler_cc_library(
    name = "IR",
    srcs = [
        "ScheduleDialect.cpp",
        "ScheduleEnums.cpp.inc",
        "ScheduleOpFolders.cpp",
        "ScheduleOpInterfaces.cpp.inc",
        "ScheduleOps.cpp",
        "ScheduleOps.cpp.inc",
        "ScheduleTypeInterfaces.cpp.inc",
        "ScheduleTypes.cpp",
        "ScheduleTypes.cpp.inc",
    ],
    hdrs = [
        "ScheduleDialect.h",
        "ScheduleEnums.h.inc",
        "ScheduleOpInterfaces.h.inc",
        "ScheduleOps.h",
        "ScheduleOps.h.inc",
        "ScheduleTraits.h",
        "ScheduleTypeInterfaces.h.inc",
        "ScheduleTypes.h",
        "ScheduleTypes.h.inc",
    ],
    deps = [
        ":ScheduleEnumsGen",
        ":ScheduleInterfacesGen",
        ":ScheduleOpsGen",
        ":ScheduleTypesGen",
        "//compiler/src/iree/compiler/Dialect/Schedule/Builtins",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ArithmeticUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

iree_gentbl_cc_library(
    name = "ScheduleEnumsGen",
    tbl_outs = [
        (
            ["--gen-enum-decls"],
            "ScheduleEnums.h.inc",
        ),
        (
            ["--gen-enum-defs"],
            "ScheduleEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ScheduleBase.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "ScheduleInterfacesGen",
    tbl_outs = [
        (
            ["--gen-op-interface-decls"],
            "ScheduleOpInterfaces.h.inc",
        ),
        (
            ["--gen-op-interface-defs"],
            "ScheduleOpInterfaces.cpp.inc",
        ),
        (
            ["--gen-type-interface-decls"],
            "ScheduleTypeInterfaces.h.inc",
        ),
        (
            ["--gen-type-interface-defs"],
            "ScheduleTypeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ScheduleInterfaces.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "ScheduleOpsGen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "ScheduleOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "ScheduleOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ScheduleOps.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "ScheduleTypesGen",
    tbl_outs = [
        (
            [
                "--gen-attrdef-decls",
                "--attrdefs-dialect=stream",
            ],
            "ScheduleAttrs.h.inc",
        ),
        (
            [
                "--gen-attrdef-defs",
                "--attrdefs-dialect=stream",
            ],
            "ScheduleAttrs.cpp.inc",
        ),
        (
            [
                "--gen-typedef-decls",
                "--typedefs-dialect=stream",
            ],
            "ScheduleTypes.h.inc",
        ),
        (
            [
                "--gen-typedef-defs",
                "--typedefs-dialect=stream",
            ],
            "ScheduleTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ScheduleBase.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "ScheduleDialectDocGen",
    tbl_outs = [
        (
            [
                "--gen-dialect-doc",
                "--dialect=stream",
            ],
            "ScheduleDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ScheduleOps.td",
    deps = [":td_files"],
)
