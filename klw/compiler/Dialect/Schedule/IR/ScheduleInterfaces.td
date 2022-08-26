// Copyright 2021 The KLW Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KLW_DIALECT_STREAM_INTERFACES
#define KLW_DIALECT_STREAM_INTERFACES

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

//===----------------------------------------------------------------------===//
// KLW::Schedule::AffinityOpInterface
//===----------------------------------------------------------------------===//

def Schedule_AffinityOp : OpInterface<"AffinityOpInterface"> {
  let description = [{
    TBD. Used to denote a stream affinity for ops and specify the kind of
    environment the ops are expected run in.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the stream affinity for the op, indicating where it should run.
      }],
      /*retTy=*/"KLW::Schedule::AffinityAttr",
      /*methodName=*/"getAffinity",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_self->getAttr("affinity").template dyn_cast_or_null<KLW::Schedule::AffinityAttr>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets the stream affinity for the op, indicating where it should run.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setAffinity",
      /*args=*/(ins "KLW::Schedule::AffinityAttr":$value),
      /*methodBody=*/[{
        $_self->setAttr("affinity", value);
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// KLW::Schedule::ScheduleableOpInterface
//===----------------------------------------------------------------------===//

def Schedule_ScheduleableOp : OpInterface<"ScheduleableOpInterface"> {
  let description = [{
    Interface for ops that can be asynchronous executed in a streaming context.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the op is slicing out/in memory instead of real work.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isMetadata",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the op should be cloned into consumer streams.
        These ops should be cheaper to recompute than to transfer their contents
        across streams (such as splats).
      }],
      /*retTy=*/"bool",
      /*methodName=*/"preferCloneToConsumers",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// KLW::Schedule::SubviewEffectOpInterface
//===----------------------------------------------------------------------===//

def Schedule_SubviewEffectOp : OpInterface<"SubviewEffectOpInterface"> {
  let description = [{
    Interface for ops that operate on subviews of resources used to query the
    memory effects for subviews on operands.
  }];

  let methods = [
    // TODO(benvanik): get memory effect + range of an operand
  ];
}

//===----------------------------------------------------------------------===//
// KLW::Schedule::BuiltinOpInterface
//===----------------------------------------------------------------------===//

def Schedule_BuiltinOp : OpInterface<"BuiltinOpInterface"> {
  let description = [{
    Interface for ops that can be asynchronous executed in a streaming context.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Merges the builtin module for the op into the target user module. By
        doing a top level module merge the builtin can provide host functions
        and multiple executables as needed.

        This should not specialize the executable yet as callers currently
        assume all builtin ops of the same type will have the same module IR.
        If we want to specialize the executables based on usage we'll need a
        more complex interface.

        Fails if there's a name conflict; we have a __ prefix and things outside
        the compiler shouldn't use it.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"mergeBuiltinModule",
      /*args=*/(ins "Operation *":$targetOp, "OpBuilder &":$targetBuilder)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Converts a stream.builtin.* op to its stream.async.* form. Any number of
        ops may be inserted such as transfers and resource operations.

        This is called in the stream.async.* phase of lowering and before
        resource refinement or copy-on-write materialization has happened; this
        enables the conversion to create new tensors as `!stream.resource<*>`.

        The original op must remain valid after the call completes; the caller
        of this needs to erase it themselves.
      }],
      /*retTy=*/"LogicalResult",
      /*methodName=*/"convertBuiltinOp",
      /*args=*/(ins "OpBuilder &":$builder)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// KLW::Schedule::TimelineOpInterface
//===----------------------------------------------------------------------===//

def Schedule_TimelineOp : OpInterface<"TimelineOpInterface"> {
  let description = [{
    Interface for ops that operate in an ordered sequence defined by timepoints.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns zero or more timepoints consumed by this timeline operation
        indicating the asynchronous operations that must complete before it can
        perform its operation.
      }],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"getAwaitTimepoints",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the timepoint produced by this timeline operation indicating the
        asynchronous completion of the operation.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getResultTimepoint",
      /*args=*/(ins)
    >,
  ];
}

#endif  // KLW_DIALECT_STREAM_INTERFACES
