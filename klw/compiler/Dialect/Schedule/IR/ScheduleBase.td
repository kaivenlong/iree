// Copyright 2021 The KLW Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KLW_DIALECT_STREAM_BASE
#define KLW_DIALECT_STREAM_BASE

include "klw/compiler/Dialect/Schedule/IR/ScheduleInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SubElementInterfaces.td"

//===----------------------------------------------------------------------===//
// KLW stream modeling dialect
//===----------------------------------------------------------------------===//

def Schedule_Dialect : Dialect {
  let name = "schedule";
  let cppNamespace = "::mlir::klw_compiler::KLW::Schedule";
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;

  let summary = [{
    A dialect designed to model execution partitioning and scheduling.
  }];
  let description = [{
    The stream dialect is designed to take tensor programs and convert them to
    explicitly scheduled asynchronous programs. This includes placing ops on
    specific targets, partitioning the work between the targets, scheduling the
    work for concurrency, and encoding tensors into target-specific resources.

    ```
    +--------+    +----------+    +-------+
    | flow.* | -> | stream.* | -> | hal.* |
    +--------+    +----------+    +-------+
    ```

    This sits in-between the `flow` and `hal` dialects.

    * `flow` models tensor programs by separating work into dispatchable
      functions in order to isolate the main host program data flow and the
      dense tensor compute operations.

    * `stream` models explicitly scheduled asynchronous programs by partitioning
      the dispatchable work, specifying target affinities, encoding tensors into
      target-specific forms, and scheduling the work to run concurrently.

    * `hal` models a low-level hardware abstraction layer used to manage
      buffers and issue asynchronous work across a variety of device types. The
      dialect is largely 1:1 with the KLW HAL C API.

    Transforms in the dialect lower tensor values into opaque resources with the
    goal of ensuring no tensors survive in the IR. At entry `stream.tensor.*`
    ops are used to capture the source tensor encoding information (data type,
    shapes, etc) and then lowered into `stream.async.*` ops that model the
    asynchronous workloads on the opaque resources. The asynchronous operations
    are then partitioned, allocated, and scheduled for execution using the
    `stream.cmd.*` ops.

    It's intended that after transformation through the stream dialect the
    program is ready for execution on an abstract machine. At this level of
    representation buffers have still not been allocated and devices are not
    yet resolved, however the information captured in the `stream` IR allows
    such operations to be done trivially. To this end all ops carry the symbolic
    size of the resources on which they operate as well as the lifetime of the
    resources they are acting upon. This manifests in the usage of the
    `!stream.resource` type:

    ```mlir
    // Unresolved lifetime (resolved during the iree-stream-refine-usage pass):
    !stream.resource<*>
    // An externally managed value (passed in via the program API).
    !stream.resource<external>
    // A staging buffer for uploads/downloads.
    !stream.resource<staging>
    // A short-lived value that is used across streams.
    !stream.resource<transient>
    // A long-lived value that persists across streams in globals.
    !stream.resource<variable>
    // An immutable value that persists for the duration of the program.
    !stream.resource<constant>
    ```

    Operations using resources carry the size of all operand result resources:

    ```mlir
    // %update (40 bytes) is being inserted into %target (296 bytes).
    // Can be dynamic values such as those originating from dynamic dimensions.
    %13 = stream.async.update %update, %target[%c256 to %c296] :
        !stream.resource<transient>{%c40} ->
        %target as !stream.resource<transient>{%c296}
    ```

    Once all `stream.async.*` work is moved into executable regions (such as
    `stream.async.execute`) `!stream.timepoint` values are used to sequence
    the execution. These timepoints represent some point in time where all
    execution up to that timepoint has completed and any results that were
    produced by the execution are available for use. Attempting to use the
    resources before their corresponding timepoint has been reached will lead
    to undefined behavior. The benefit of this is that after timepoints are
    established in the IR it's possible to induce aliasing of resources without
    breaking execution correctness.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base stream dialect op classes
//===----------------------------------------------------------------------===//

class Schedule_Op<string mnemonic, list<Trait> traits = []> :
    Op<Schedule_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Schedule dialect types
//===----------------------------------------------------------------------===//

def Schedule_PrimitiveType : AnyTypeOf<[Index, AnyInteger, AnyFloat]>;
def Schedule_FillPatternType : AnyTypeOf<[I8, I16, I32]>;

def Schedule_Offset : TypeAlias<Index>;
def Schedule_Size : TypeAlias<Index>;

def Schedule_Tensor : TypeAlias<AnyRankedTensor>;
def Schedule_Dim : TypeAlias<Index>;
def Schedule_ShapeDynamicDims : Variadic<Schedule_Dim>;

def Schedule_IndexAttr : Util_IndexAttrBase<"index">;
def Schedule_IndexArrayAttr : TypedArrayAttrBase<Schedule_IndexAttr,
                                               "index array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

def Schedule_ExecutableRefAttr : Util_AliasedSymbolRefAttr;
def Schedule_GlobalRefAttr : Util_AliasedSymbolRefAttr;
def Schedule_GlobalPtr : Util_AnyPtrOf<[Schedule_Tensor, Schedule_PrimitiveType]>;

//===----------------------------------------------------------------------===//
// Schedule attributes
//===----------------------------------------------------------------------===//

def Schedule_AffinityAttr : AttrDef<Schedule_Dialect, "Affinity", []> {
  let mnemonic = "affinity";

  let summary = [{defines execution context affinity}];
  let description = [{
    TBD. The intent is that this can specify host, device, and queue affinity.
    Scopes can be annotated with an affinity to ensure execution within happens
    in a particular location. Arrays of affinities or wildcard specifiers will
    allow for refinement ("do it on this device but auto select a queue"). It
    will also allow us to indicate host affinity such that device<->device and
    host<->device can be identified in the IR structure. Today all affinities
    are no-op'ed and assumed to be 'current device'.
  }];

  // TODO(benvanik): affinity.
  let parameters = (ins);

  let valueType = NoneType;

  let extraClassDeclaration = [{
    // Returns an affinity active for the given operation.
    // This will recursively walk parent operations until one with the
    // `stream.affinity` attribute is found.
    static AffinityAttr lookup(Operation *op);

    // Returns true if |desiredAffinity| (if any) is compatible with
    // |requiredAffinity|.
    static bool areCompatible(AffinityAttr desiredAffinity,
                              AffinityAttr requiredAffinity);
  }];
}

def Schedule_Favor_Debug : I32EnumAttrCase<"Debug", 0, "debug">;
def Schedule_Favor_MinPeakMemory : I32EnumAttrCase<"MinPeakMemory", 1, "min-peak-memory">;
def Schedule_Favor_MaxConcurrency : I32EnumAttrCase<"MaxConcurrency", 2, "max-concurrency">;
def Schedule_FavorAttr :
    I32EnumAttr<"Favor", "KLW partitioning bias", [
      Schedule_Favor_Debug,
      Schedule_Favor_MinPeakMemory,
      Schedule_Favor_MaxConcurrency,
    ]> {
  let cppNamespace = "::mlir::klw_compiler::KLW::Schedule";
}

def Schedule_PartitioningConfigAttr :
    AttrDef<Schedule_Dialect, "PartitioningConfig", [
      DeclareAttrInterfaceMethods<SubElementAttrInterface>,
    ]> {
  let mnemonic = "partitioning_config";
  let summary = [{defines partitioning configuration}];
  let description = [{
    Configures the partitioning algorithm to use and its configuration.
    Partitioning is useful to adjust when scheduling behavior of targets is
    radically different - such as single-threaded vs. multi-threaded CPUs or
    bespoke ML accelerators vs. general purpose GPUs. This mechanism controls
    the amount of concurrency, parallelism, memory consumption, and latency.
  }];

  // TODO(benvanik): partitioning config.
  let parameters = (ins
    "KLW::Schedule::FavorAttr":$favor
  );

  let valueType = NoneType;

  let builders = [
    AttrBuilderWithInferredContext<(ins "KLW::Schedule::FavorAttr":$favor), [{
      return $_get(favor.getContext(), favor);
    }]>,
  ];

  let extraClassDeclaration = [{
    // Returns a partitioning config active for the given operation.
    // This will recursively walk parent operations until one with the
    // `stream.partitioning` attribute is found.
    static PartitioningConfigAttr lookup(Operation *op);
  }];

  let hasCustomAssemblyFormat = 1;
}

def Schedule_ResourceConfigAttr :
    AttrDef<Schedule_Dialect, "ResourceConfig", []> {
  let mnemonic = "resource_config";

  let summary = [{defines resource constraints configuration}];
  let description = [{
    Defines resource storage constraints. These allow for packing and layout
    algorithms to ensure they are producing usable results on target devices.
  }];

  // TODO(benvanik): this was just copied from the HAL; probably good to make it
  // more generic such that we can classify entire device families instead of
  // specific values like this. As-is this is a phase-ordering violation.
  let parameters = (ins
    // The maximum size of a memory allocation that can be created, even if
    // there is more space available in the heap.
    "int64_t":$maxAllocationSize,
    // The minimum required alignment, in bytes, for offsets used in runtime
    // resource bindings. Offset values (both dynamic and static) must be an
    // integer multiple of this limit.
    "int64_t":$minBufferOffsetAlignment,
    // The maximum value that can be specified for size ranges of resource
    // bindings. The underlying allocation may be larger than this but only
    // up to this amount will be visible to kernels.
    "int64_t":$maxBufferRange,
    // The minimum required alignment, in bytes, for size ranges of resources
    // bindings.
    "int64_t":$minBufferRangeAlignment,
    // Number of bits in `index` values as passed across device boundaries.
    "int64_t":$indexBits
  );

  let valueType = NoneType;

  let extraClassDeclaration = [{
    // Returns the intersection (most conservative) constraints |lhs| ∩ |rhs|.
    static ResourceConfigAttr
    intersectBufferConstraints(ResourceConfigAttr lhs, ResourceConfigAttr rhs);

    // Returns a resource config compatible with the host.
    // These must only be used with resources when it is known that the device
    // is local or has unified memory.
    static ResourceConfigAttr
    getDefaultHostConstraints(MLIRContext *context);

    // Returns a resource config active for the given operation.
    // This will recursively walk parent operations until one with the
    // `stream.resources` attribute is found, an affinity specifies a
    // configuration, or as a fallback returns a conservative configuration.
    static ResourceConfigAttr lookup(Operation *op);
  }];

  let hasCustomAssemblyFormat = 1;
}

def Schedule_ResourceAccess_None : I32BitEnumAttrCase<"None", 0x0000>;
def Schedule_ResourceAccess_Read : I32BitEnumAttrCase<"Read", 0x0001>;
def Schedule_ResourceAccess_Write : I32BitEnumAttrCase<"Write", 0x0002>;
def Schedule_ResourceAccessBitfieldAttr :
    I32BitEnumAttr<"ResourceAccessBitfield", "valid ResourceAccess", [
      Schedule_ResourceAccess_None,
      Schedule_ResourceAccess_Read,
      Schedule_ResourceAccess_Write,
    ]> {
  let cppNamespace = "mlir::klw_compiler::KLW::Schedule";
}
def Schedule_ResourceAccessArrayAttr :
    TypedArrayAttrBase<Schedule_ResourceAccessBitfieldAttr,
                       "access array attribute"> {}

//===----------------------------------------------------------------------===//
// Schedule synchronization types
//===----------------------------------------------------------------------===//

def Schedule_Timepoint : TypeDef<Schedule_Dialect, "Timepoint", [
  Util_GlobalType,
]> {
  let mnemonic = "timepoint";

  let summary = [{a timepoint indicating execution availability}];
  let description = [{
    Represents a point in the execution timeline that when resolved indicates
    that all of the execution prior to this timepoint has completed and the
    results of the execution are available for use. This includes transitive
    dependencies as well; if timepoint B is dependent on timepoint A then when
    B is available so too must be A.
  }];

  // TODO(benvanik): track affinity so we know where timepoints come from.
  let parameters = (ins);
}

def Schedule_TimepointAttr : AttrDef<Schedule_Dialect, "Timepoint",
                             [TypedAttrInterface]> {
  let mnemonic = "timepoint";
  let summary = [{an immediately-resolved timepoint}];
  let description = [{}];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let valueType = Schedule_Timepoint;
  let constBuilderCall = [{
    KLW::Schedule::TimepointAttr::get(
        $_builder.getContext(),
        KLW::Schedule::TimepointType::get($_builder.getContext()));
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Schedule resource value types
//===----------------------------------------------------------------------===//

// Indicates a value whose lifetime is not yet analyzed.
def Schedule_Lifetime_Unknown : I32EnumAttrCase<"Unknown", 0, "*">;
// An externally managed value.
def Schedule_Lifetime_External : I32EnumAttrCase<"External", 1, "external">;
// A staging buffer for uploads/downloads.
def Schedule_Lifetime_Staging : I32EnumAttrCase<"Staging", 2, "staging">;
// A short-lived value that is used across streams.
def Schedule_Lifetime_Transient : I32EnumAttrCase<"Transient", 3, "transient">;
// A long-lived value that persists across streams.
def Schedule_Lifetime_Variable : I32EnumAttrCase<"Variable", 4, "variable">;
// An immutable value that persist for the duration of the program.
def Schedule_Lifetime_Constant : I32EnumAttrCase<"Constant", 5, "constant">;
def Schedule_LifetimeAttr :
    I32EnumAttr<"Lifetime", "KLW Schedule value lifetime", [
      Schedule_Lifetime_Unknown,
      Schedule_Lifetime_External,
      Schedule_Lifetime_Staging,
      Schedule_Lifetime_Transient,
      Schedule_Lifetime_Variable,
      Schedule_Lifetime_Constant,
    ]> {
  let cppNamespace = "::mlir::klw_compiler::KLW::Schedule";
}

def Schedule_AnyResource : Type<
    CPred<"$_self.isa<KLW::Schedule::ResourceType>()">,
    "any stream-compatible type">;

// TODO(benvanik): support other types; the interface may be enough.
def Schedule_TransferType : AnyTypeOf<[
  Schedule_AnyResource,
  Schedule_Tensor,
  Schedule_PrimitiveType,
]>;

def Schedule_Resource : TypeDef<Schedule_Dialect, "Resource", [
  Util_ReferenceType,
  Util_SizeAwareType,
  DeclareTypeInterfaceMethods<Util_GlobalType, [
    "isAccessStorageCompatible",
  ]>,
  DeclareTypeInterfaceMethods<Util_InferTypeSize, [
    "inferSizeFromValue",
  ]>,
  DeclareTypeInterfaceMethods<Util_SubrangeType, [
    "createSubrangeOp",
  ]>,
]> {
  let mnemonic = "resource";

  let summary = [{a managed resource}];
  let description = [{
    Schedule external values represent asynchronously-available and sequenced
    values that are owned and managed by external code - such as those passed in
    or out of the program entry points. Though external values are managed
    during an invocation the same as other stream values the visibility into
    them does not extend outside of the invocation they are provided to.

    Schedule values are not usable directly outside of a stream execution or
    transfer operation. If the contents of the value are needed they must first
    be transferred via `stream.transfer` - which may incur a copy.
  }];

  let parameters = (ins
    "KLW::Schedule::Lifetime":$lifetime
  );

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, KLW::Schedule::Lifetime::Unknown);
    }]>,
    TypeBuilderWithInferredContext<(ins
      "KLW::Schedule::LifetimeAttr":$lifetime
    ), [{
      return $_get(lifetime.getContext(), lifetime.getValue());
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;
}

def Schedule_ResourceLifetimeUnknown : CPred<[{
  $_self.cast<KLW::Schedule::ResourceType>().getLifetime() ==
      KLW::Schedule::Lifetime::Unknown
}]>;
def Schedule_ResourceLifetimeExternal : CPred<[{
  $_self.cast<KLW::Schedule::ResourceType>().getLifetime() ==
      KLW::Schedule::Lifetime::External
}]>;
def Schedule_ResourceLifetimeStaging : CPred<[{
  $_self.cast<KLW::Schedule::ResourceType>().getLifetime() ==
      KLW::Schedule::Lifetime::Staging
}]>;
def Schedule_ResourceLifetimeTransient : CPred<[{
  $_self.cast<KLW::Schedule::ResourceType>().getLifetime() ==
      KLW::Schedule::Lifetime::Transient
}]>;
def Schedule_ResourceLifetimeVariable : CPred<[{
  $_self.cast<KLW::Schedule::ResourceType>().getLifetime() ==
      KLW::Schedule::Lifetime::Variable
}]>;
def Schedule_ResourceLifetimeConstant : CPred<[{
  $_self.cast<KLW::Schedule::ResourceType>().getLifetime() ==
      KLW::Schedule::Lifetime::Constant
}]>;

def Schedule_UnknownResource : DialectType<Schedule_Dialect, And<[
  CPred<"$_self.isa<KLW::Schedule::ResourceType>()">,
  Schedule_ResourceLifetimeUnknown,
]>, "resource"> {
  let description = [{
    A stream resource that has not yet had its lifetime calculated.
  }];
}

def Schedule_ExternalResource : DialectType<Schedule_Dialect, And<[
  CPred<"$_self.isa<KLW::Schedule::ResourceType>()">,
  Schedule_ResourceLifetimeExternal,
]>, "external resource"> {
  let description = [{
    Schedule external values represent asynchronously-available and sequenced
    values that are owned and managed by external code - such as those passed in
    or out of the program entry points. Though external values are managed
    during an invocation the same as other stream values the visibility into
    them does not extend outside of the invocation they are provided to.

    Schedule values are not usable directly outside of a stream execution or
    transfer operation. If the contents of the value are needed they must first
    be transferred via `stream.transfer` - which may incur a copy.
  }];
}

def Schedule_StagingResource : DialectType<Schedule_Dialect, And<[
  CPred<"$_self.isa<KLW::Schedule::ResourceType>()">,
  Schedule_ResourceLifetimeStaging,
]>, "staging resource"> {
  let description = [{
    Schedule upload/download staging resource. These are used outside of streams
    and then transferred to other stream resources such as variables or
    transients for use inside of streams. Dispatches and several other
    operations cannot directly operate on these resources.
  }];
}

def Schedule_TransientResource : DialectType<Schedule_Dialect, And<[
  CPred<"$_self.isa<KLW::Schedule::ResourceType>()">,
  Schedule_ResourceLifetimeTransient,
]>, "transient resource"> {
  let description = [{
    Schedule transients represent asynchronously-available and sequenced values
    that have a short lifetime - often only passed between stream executions.
    It is expected that transient values are not stored in global state and have
    minimal lifetime as they may be heavily pooled or suballocated.

    Schedule values are not usable directly outside of a stream execution or
    transfer operation. If the contents of the value are needed they must first
    be transferred via `stream.transfer` - which may incur a copy.
  }];
}

def Schedule_VariableResource : DialectType<Schedule_Dialect, And<[
  CPred<"$_self.isa<KLW::Schedule::ResourceType>()">,
  Schedule_ResourceLifetimeVariable,
]>, "variable resource"> {
  let description = [{
    Schedule variables represent asynchronously-available and sequenced values
    that have a long lifetime relative to the work being performed on them.
    These variables are often stored in global state and may live for the entire
    duration of the program.

    Schedule values are not usable directly outside of a stream execution or
    transfer operation. If the contents of the value are needed they must first
    be transferred via `stream.transfer` - which may incur a copy.
  }];
}

def Schedule_ConstantResource : DialectType<Schedule_Dialect, And<[
  CPred<"$_self.isa<KLW::Schedule::ResourceType>()">,
  Schedule_ResourceLifetimeConstant,
]>, "constant resource"> {
  let description = [{
    Schedule constants are immutable values that are available for the lifetime of
    the program once initialized.
  }];
}

def Schedule_AnyScheduleResource : AnyTypeOf<[
  Schedule_UnknownResource,
  Schedule_ExternalResource,
  Schedule_TransientResource,
  Schedule_VariableResource,
  Schedule_ConstantResource,
]>;

//===----------------------------------------------------------------------===//
// Executable bindings
//===----------------------------------------------------------------------===//

def Schedule_Binding : TypeDef<Schedule_Dialect, "Binding", []> {
  let mnemonic = "binding";

  let summary = [{a managed resource binding into an executable scope}];
  let description = [{
    A resource binding available within an executable dispatch function.
    The bindings map 1:1 with the resources bound during dispatch operations.
  }];

  // TODO(benvanik): carry lifetime like resources.
  let parameters = (ins);

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt);
    }]>,
  ];
}

def Schedule_AnyBinding : AnyTypeOf<[
  Schedule_Binding,
]>;

//===----------------------------------------------------------------------===//
// Schedule op traits
//===----------------------------------------------------------------------===//

def Schedule_TensorPhaseOp : NativeOpTrait<"KLW::Schedule::TensorPhaseOp">;
def Schedule_AsyncPhaseOp : NativeOpTrait<"KLW::Schedule::AsyncPhaseOp">;
def Schedule_CmdPhaseOp : NativeOpTrait<"KLW::Schedule::CmdPhaseOp">;

#endif  // KLW_DIALECT_STREAM_BASE
