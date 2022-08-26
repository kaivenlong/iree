// Copyright 2021 The KLW Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KLW_DIALECT_SCHEDULE_OPS
#define KLW_DIALECT_SCHEDULE_OPS

include "klw/compiler/Dialect/Schedule/IR/ScheduleBase.td"
include "klw/compiler/Dialect/Schedule/IR/ScheduleInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class Schedule_PureOp<string mnemonic, list<Trait> traits = []> :
    Schedule_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// Generic resource ops
//===----------------------------------------------------------------------===//

def Schedule_ResourceAllocOp : Schedule_PureOp<"resource.alloc", [
  DeclareOpInterfaceMethods<Schedule_AffinityOp, [
    "getAffinity",
    "setAffinity",
  ]>,
  Util_SizeAwareOp,
  MemoryEffects<[MemAlloc]>,
]> {
  let summary = [{allocates a persistent value with undefined contents}];
  let description = [{
    Allocates a persistent value (one that is long-lived and possibly external
    to the program) with undefined contents. Consumers of the allocated
    result must assume nothing of the contents and use `discard` access.

    Uninitialized allocations will have undefined contents and must only be used
    when all bytes are discarded prior to any reads. Runtimes decide what
    "undefined contents" means and here it only indicates that execution will be
    correct even if the memory starts with non-zero values.

    If multiple values are allocated from the same operation it implies that
    they have matching lifetimes. When lowering to execution environments the
    separate allocations may be fused into one or more slab allocations in order
    to reduce overheads. How many allocations can be fused is based on the size
    of the individual resources and the target constraints (how large any single
    buffer may be, etc). At the stream dialect level treat a multi-result alloc
    as a way to indicate similar lifetimes.
  }];

  let arguments = (ins
    Variadic<Schedule_Size>:$storage_sizes,
    UnitAttr:$uninitialized,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<Schedule_AnyResource>:$results
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    (`uninitialized` $uninitialized^)?
    attr-dict `:` custom<SizeAwareTypeList>(type($results), $storage_sizes)
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getStorageSizes()[idx]; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_ResourceAllocaOp : Schedule_PureOp<"resource.alloca", [
  DeclareOpInterfaceMethods<Schedule_AffinityOp, [
    "getAffinity",
    "setAffinity",
  ]>,
  Schedule_TimelineOp,
  Util_SizeAwareOp,
  MemoryEffects<[MemAlloc]>,
]> {
  let summary = [{allocates a transient value with undefined contents}];
  let description = [{
    Allocates a transient value (one that is short-lived and local to the
    current computation) with undefined contents. Consumers of the allocated
    result must assume nothing of the contents and use `discard` access.

    The resource returned is not valid for use until the timepoint is reached;
    execution using this resource must await on the timepoint.
  }];

  let arguments = (ins
    Schedule_Size:$storage_size,
    Optional<Schedule_Timepoint>:$await_timepoint,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    AnyTypeOf<[
      Schedule_StagingResource,
      Schedule_TransientResource,
    ]>:$result,
    Schedule_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    `uninitialized`
    (`on` `(` $affinity^ `)`)?
    (`await` `(` $await_timepoint^ `)` `=` `` `>`):(`:`)?
    attr-dict
    type($result) `{` $storage_size `}`
    `=` `` `>`
    type($result_timepoint)
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getStorageSize(); }
    SmallVector<Value> getAwaitTimepoints() {
      if (getAwaitTimepoint()) return {getAwaitTimepoint()}; else return {};
    }
  }];

  let hasCanonicalizer = 1;
}

def Schedule_ResourceDeallocaOp : Schedule_Op<"resource.dealloca", [
  DeclareOpInterfaceMethods<Schedule_AffinityOp, [
    "getAffinity",
    "setAffinity",
  ]>,
  Schedule_TimelineOp,
  Util_SizeAwareOp,
  MemoryEffects<[MemFree]>,
]> {
  let summary = [{frees a transient value when available}];
  let description = [{
    Deallocates a transient value (one that is short-lived and local to the
    current computation) previously allocated using `stream.resource.alloca`.

    The resource is considered live and valid until the provided timepoint is
    reached and the memory is only made available for future requests after
    the result timepoint is reached.
  }];

  let arguments = (ins
    AnyTypeOf<[
      Schedule_StagingResource,
      Schedule_TransientResource,
    ]>:$operand,
    Schedule_Size:$operand_size,
    Optional<Schedule_Timepoint>:$await_timepoint,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    (`await` `(` $await_timepoint^ `)` `=` `` `>`)?
    $operand `:` type($operand) `{` $operand_size `}`
    `=` `` `>` type($result_timepoint)
    attr-dict
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getOperandSize(); }
    Value getResultSize(unsigned idx) { return {}; }
    SmallVector<Value> getAwaitTimepoints() {
      if (getAwaitTimepoint()) return {getAwaitTimepoint()}; else return {};
    }
  }];

  let hasCanonicalizer = 1;
}

def Schedule_ResourceSizeOp : Schedule_PureOp<"resource.size", [
  Schedule_AffinityOp,
  Util_SizeAwareOp,
]> {
  let summary = [{returns the size of the resource storage in bytes}];
  let description = [{
    Returns a possibly runtime-dynamic byte size of the resource backing
    storage. This may differ from the logical storage size of a value based on
    the alignment requirements of the target as well as encoding of higher level
    values such as sparse tensor formats.
  }];

  let arguments = (ins
    Schedule_AnyResource:$operand,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_Size:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $operand
    attr-dict `:` type($operand)
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getResult(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Schedule_ResourceMapOp : Schedule_Op<"resource.map", [
  Schedule_AffinityOp,
  Util_SizeAwareOp,
  MemoryEffects<[MemAlloc]>,
]> {
  let summary = [{maps read-only memory into a staging resource}];
  let description = [{
    Synchronously maps a host heap buffer into a stream-accessible staging
    resource. Will never fail but may induce a copy if required and as such the
    mapped resource is not coherent with the original source buffer: changing
    the source buffer after mapping has undefined behavior.
  }];

  let arguments = (ins
    Util_BufferType:$source,
    Schedule_Offset:$source_offset,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_StagingResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `[` $source_offset `]` `:`
    type($source)
    `->`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_ResourceTryMapOp : Schedule_PureOp<"resource.try_map", [
  Schedule_AffinityOp,
  Util_SizeAwareOp,
  MemoryEffects<[MemAlloc]>,
]> {
  let summary = [{maps read-only memory into a resource}];
  let description = [{
    Synchronously maps a host heap buffer into a stream-accessible resource
    with constant lifetime. If the given source cannot be mapped into a constant
    a failure is returned and the resulting resource value is null. As with
    `stream.resource.map` the resulting resource is not coherent with the source
    and changes will not be reflected.
  }];

  let arguments = (ins
    Util_BufferType:$source,
    Schedule_Offset:$source_offset,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    I1:$did_map,
    Schedule_ConstantResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `[` $source_offset `]` `:`
    type($source)
    `->`
    type($did_map) `,` type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_ResourceLoadOp : Schedule_Op<"resource.load", [
  Util_SizeAwareOp,
]> {
  let summary = [{loads a value from a staging resource}];
  let description = [{
    Returns the element(s) at the given offset in the staging resource.
    The operation will complete synchronously against the resource though it may
    introduce a yield point if the staging resource needs to be transferred.
  }];

  let arguments = (ins
    Schedule_StagingResource:$source,
    Schedule_Size:$source_size,
    Schedule_Offset:$source_offset
  );
  let results = (outs
    AnyTypeOf<[Schedule_PrimitiveType, AnyVector]>:$result
  );

  let assemblyFormat = [{
    $source `[` $source_offset `]` `:`
    type($source) `` `{` $source_size `}`
    `->`
    type($result)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_ResourceStoreOp : Schedule_Op<"resource.store", [
  Util_SizeAwareOp,
  MemoryEffects<[MemWrite]>,
]> {
  let summary = [{stores a value into a staging resource}];
  let description = [{
    The operation will complete synchronously against the resource though it may
    introduce a yield point if the staging resource needs to be acquired.
  }];

  let arguments = (ins
    Schedule_StagingResource:$target,
    Schedule_Size:$target_size,
    Schedule_Offset:$target_offset,
    AnyTypeOf<[Schedule_PrimitiveType, AnyVector]>:$value
  );

  let assemblyFormat = [{
    $value `,`
    $target `[` $target_offset `]` `:`
    type($value)
    `->`
    type($target) `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_ResourcePackOp : Schedule_PureOp<"resource.pack", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  Schedule_AffinityOp,
]> {
  let summary = [{packs variable-sized slices into a single slab}];
  let description = [{
    Performs a greedy packing of one or more sized slices with specified
    lifetimes and returns their relative offsets in an aliased linear space.

    Slices are `[start, end] = %slice_byte_size`, where the start and end values
    define an inclusive lifetime range and the size is the total number of bytes
    required to be live for that range.

    ```mlir
    // Computes the total length required for the packed values and the offsets
    // of the 3 slices requested relative to the base of the packed memory:
    %total_length, %offset_0, %offset_1, %offset_2 =
        stream.resource.pack
            // Each slice gets one result offset:
            slices({
              // 3 slices where A and B overlap and will get unique offsets
              // while B and C do not overlap and are allowed to alias.
              [0, 10] = %size_0,  // A => %offset_0
              [3,  8] = %size_1,  // B => %offset_1
              [9, 10] = %size_2,  // C => %offset_2
              ...
            }) : index
    ```

    The lifetime start and end points (inclusive) are only used for relative
    comparisons and may originate with any meaning (op order in block, epoch,
    phase of the moon, etc). The packing algorithm uses the intervals to
    determine slice liveness and when aliasing is safe.

    The size of each slice may either be a constant or runtime-computed dynamic
    value. Constant slices can achieve more dense packing than the dynamic
    values and CSE/canonicalization should be applied to ensure that as many of
    the dynamic values are equivalent if possible.

    The total length required to pack all slices is returned and can be used to
    acquire storage. The individual slice offsets are 0-based and as such if are
    directly used as buffer offsets may need additional offsetting. This can
    either be applied via the optional `offset` operand or slicing of the
    underlying allocation buffer.
  }];

  let arguments = (ins
    Optional<Schedule_Offset>:$offset,
    Schedule_IndexArrayAttr:$lifetime_intervals,
    Variadic<Schedule_Size>:$dynamic_slice_sizes,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_Size:$total_length,
    Variadic<Schedule_Offset>:$packed_offsets
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    (`offset` `(` $offset^ `)`)?
    `slices` `(` `{`
    custom<PackSliceRanges>($lifetime_intervals,
                            $dynamic_slice_sizes,
                            type($packed_offsets))
    `}` `)`
    `:` type($total_length)
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    struct Slice {
      int64_t lifetimeStart;
      int64_t lifetimeEnd;
      Value dynamicSize;
      Value packedOffset;

      bool operator==(const Slice &rhs) const {
        return lifetimeStart == rhs.lifetimeStart &&
               lifetimeEnd == rhs.lifetimeEnd;
      }
      bool operator!=(const Slice &rhs) const {
        return !(*this == rhs);
      }
      bool operator<(const Slice &rhs) const {
        return std::make_pair(lifetimeStart, lifetimeEnd) <
               std::make_pair(rhs.lifetimeStart, rhs.lifetimeEnd);
      }
      bool intersects(const Slice &rhs) const {
        return lifetimeEnd >= rhs.lifetimeStart &&
               rhs.lifetimeEnd >= lifetimeStart;
      }
    };

    /// Returns all of the slices to be packed.
    /// Order is ascending by lifetime interval (post-canonicalization).
    SmallVector<Slice> getSlices();
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Schedule_ResourceConstantsOp : Schedule_PureOp<"resource.constants", [
  SameVariadicResultSize,
  Schedule_AffinityOp,
  Schedule_TimelineOp,
  Util_SizeAwareOp,
]> {
  let summary = [{asynchronously uploads or maps constant values}];
  let description = [{
    Represents an upload of constant resources that may be packed, suballocated,
    and mapped depending on the final lowering target.

    In runtime environments where memory is shared between host and device this
    turns into a mapping operation that avoids additional memory allocation and
    copies. When memory cannot be shared an asynchronous stream will be created
    to allocate and copy all of the constant values.

    Though this op returns a unique resource for each constant value it's
    expected that almost all end up aliasing into the same storage. The exact
    packing and number of storage resources that are needed are not known until
    lowering to a particular backend, though, so they are separate here for
    proper usage tracking.

    Both constant and variable resources can be produced; a constant is
    immutable while a variable will be treated as a constant-value initializer
    for a mutable resource. By modeling these together it's not required that
    variable initializers first be allocated, copied to the target, and then
    copied into the variable storage if the target is capable of doing a direct
    upload or mapping.
  }];

  let arguments = (ins
    TypedArrayAttrBase<ElementsAttr, "constant value array attribute">:$values,
    Variadic<Schedule_Size>:$result_sizes,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<AnyTypeOf<[
      Schedule_ConstantResource,
      Schedule_VariableResource,
    ]>>:$results,
    Schedule_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    attr-dict `:`
    custom<ConstantValueList>(type($results),
                              $result_sizes,
                              $values)
    `\n` ` ` ` ` `=` `` `>` type($result_timepoint)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSizes()[idx]; }
    SmallVector<Value> getAwaitTimepoints() { return {}; }
  }];
}

def Schedule_ResourceSubviewOp : Schedule_PureOp<"resource.subview", [
  AllTypesMatch<["source", "result"]>,
  DeclareOpInterfaceMethods<ViewLikeOpInterface>,
  DeclareOpInterfaceMethods<Schedule_ScheduleableOp, [
    "isMetadata",
  ]>,
  Util_SizeAwareOp,
  Util_SubrangeOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{slices out a cloned subview of a value}];
  let description = [{
    Aliases a byte subrange of a resource.
  }];

  let arguments = (ins
    Schedule_AnyResource:$source,
    Schedule_Size:$source_size,
    Schedule_Offset:$source_offset,
    Schedule_Size:$result_size
  );
  let results = (outs
    Schedule_AnyResource:$result
  );

  let assemblyFormat = [{
    $source `[` $source_offset `]` `:`
    type($source) `` `{` $source_size `}` `->`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }

    Value getSubrangeResource() { return getSource(); }
    Value getSubrangeResourceSize() { return getSourceSize(); }
    Value getSubrangeOffset() { return getSourceOffset(); }
    Value getSubrangeLength() { return getResultSize(); }
    Value getSubrangeResult() { return getResult(); }

    // Walks up the use-def chain to find a subview op that feeds into |value|.
    static KLW::Schedule::ResourceSubviewOp findSubviewOp(Value value);
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Pseudo ops for conversion support
//===----------------------------------------------------------------------===//

def Schedule_TensorImportOp : Schedule_PureOp<"tensor.import", [
  Schedule_AffinityOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{conversion placeholder for other->stream type conversion}];
  let description = [{
    Defines a conversion from a higher-level dialect type such as `tensor` that
    is resolved during lowering into the stream dialect. This can be used to
    interoperate between levels of the stack that require specifying stream
    types and those that prior to lowering do not handle them.
  }];

  let arguments = (ins
    AnyType:$source,
    TypeAttr:$result_encoding,
    Schedule_ShapeDynamicDims:$result_encoding_dims,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_ExternalResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `:`
    type($source)
    `->`
    $result_encoding (`` `{` $result_encoding_dims^ `}`)?
    `in`
    type($result) `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultEncodingDims(); }
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Schedule_TensorExportOp : Schedule_PureOp<"tensor.export", [
  Schedule_AffinityOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{conversion placeholder for stream->other type conversion}];
  let description = [{
    Defines a conversion to a higher-level dialect type such as `tensor` that
    is resolved during lowering into the stream dialect. This can be used to
    interoperate between levels of the stack that require specifying stream
    types and those that prior to lowering do not handle them.
  }];

  let arguments = (ins
    Schedule_ExternalResource:$source,
    TypeAttr:$source_encoding,
    Schedule_ShapeDynamicDims:$source_encoding_dims,
    Schedule_Size:$source_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `:`
    $source_encoding (`` `{` $source_encoding_dims^ `}`)?
    `in`
    type($source) `` `{` $source_size `}`
    `->`
    type($result)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceEncodingDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return ValueRange{}; }
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// High-level tensor ops
//===----------------------------------------------------------------------===//

def Schedule_TensorSizeOfOp : Schedule_PureOp<"tensor.sizeof", [
  Schedule_AffinityOp,
  Schedule_TensorPhaseOp,
]> {
  let summary = [{calculates the storage size of a given high-level type}];
  let description = [{
    Target-dependent storage size calculation using a high-level annotated type.
    While within the stream dialect the storage size of a value is left as a
    placeholder using this op. The requisite target-specific parameters for
    expanding the size calculation are only available after affinities have been
    assigned.
  }];

  let arguments = (ins
    TypeAttr:$encoding,
    Schedule_ShapeDynamicDims:$encoding_dims,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_Size:$storage_size
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $encoding (`{` $encoding_dims^ `}`)?
    attr-dict `:` type($storage_size)
  }];

  let hasVerifier = 1;
}

def Schedule_TensorEmptyOp : Schedule_PureOp<"tensor.empty", [
  Schedule_AffinityOp,
  Schedule_ScheduleableOp,
  Schedule_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{defines an empty tensor value}];
  let description = [{
    Returns a typed resource initialized with no contents. This still carries
    shape metadata and may encode to a non-empty resource such as in cases
    where the empty representation still has data (e.g. sparse tensors).
  }];

  let arguments = (ins
    TypeAttr:$result_encoding,
    Schedule_ShapeDynamicDims:$result_encoding_dims,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    `:`
    $result_encoding (`` `{` $result_encoding_dims^ `}`)?
    `in`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultEncodingDims(); }
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;
}

def Schedule_TensorConstantOp : Schedule_PureOp<"tensor.constant", [
  Schedule_AffinityOp,
  Schedule_ScheduleableOp,
  Schedule_TensorPhaseOp,
  Util_ShapeAwareOp,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{defines a constant tensor value}];
  let description = [{
    Returns a typed resource initialized to the given constant value.
  }];

  let arguments = (ins
    ElementsAttr:$value,
    TypeAttr:$result_encoding,
    Schedule_ShapeDynamicDims:$result_encoding_dims,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    `:`
    $result_encoding (`` `{` $result_encoding_dims^ `}`)?
    `in`
    type($result)
    `=`
    $value
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultEncodingDims(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_TensorSplatOp : Schedule_PureOp<"tensor.splat", [
  Schedule_AffinityOp,
  Schedule_ScheduleableOp,
  Schedule_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
]> {
  let summary = [{splats a value into a shaped tensor}];
  let description = [{
    Returns a typed resource initialized to the given primitive value.
  }];

  let arguments = (ins
    Schedule_PrimitiveType:$value,
    TypeAttr:$result_encoding,
    Schedule_ShapeDynamicDims:$result_encoding_dims,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $value
    `:` type($value)
    `->`
    $result_encoding (`` `{` $result_encoding_dims^ `}`)?
    `in`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultEncodingDims(); }
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_TensorCloneOp : Schedule_PureOp<"tensor.clone", [
  AttrSizedOperandSegments,
  Schedule_AffinityOp,
  Schedule_ScheduleableOp,
  Schedule_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
]> {
  let summary = [{clones the contents of a value}];
  let description = [{
    Clones the contents of a value at a snapshot in time. Future changes to the
    cloned value will not effect the result. Acts as a copy-on-write operation.
  }];

  let arguments = (ins
    Schedule_AnyScheduleResource:$source,
    TypeAttr:$source_encoding,
    Schedule_ShapeDynamicDims:$source_encoding_dims,
    Schedule_Size:$source_size,
    TypeAttr:$result_encoding,
    Schedule_ShapeDynamicDims:$result_encoding_dims,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `:`
    $source_encoding (`` `{` $source_encoding_dims^ `}`)?
    `in`
    type($source) `` `{` $source_size `}`
    `->`
    $result_encoding (`` `{` $result_encoding_dims^ `}`)?
    `in`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceEncodingDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultEncodingDims(); }
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Schedule_TensorSliceOp : Schedule_PureOp<"tensor.slice", [
  AttrSizedOperandSegments,
  Schedule_AffinityOp,
  Schedule_ScheduleableOp,
  Schedule_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
]> {
  let summary = [{slices out a cloned subview of a value}];
  let description = [{
    Slices a subrange of a stream resource based on a tensor encoding. Acts as a
    copy-on-write operation.
  }];

  let arguments = (ins
    Schedule_AnyScheduleResource:$source,
    TypeAttr:$source_encoding,
    Schedule_ShapeDynamicDims:$source_encoding_dims,
    Schedule_Size:$source_size,
    Variadic<Schedule_Dim>:$start_indices,
    Variadic<Schedule_Dim>:$lengths,
    TypeAttr:$result_encoding,
    Schedule_ShapeDynamicDims:$result_encoding_dims,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `[` $start_indices `for` $lengths `]` `:`
    $source_encoding (`` `{` $source_encoding_dims^ `}`)?
    `in`
    type($source) `` `{` $source_size `}`
    `->`
    $result_encoding (`` `{` $result_encoding_dims^ `}`)?
    `in`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceEncodingDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultEncodingDims(); }
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Schedule_TensorFillOp : Schedule_Op<"tensor.fill", [
  AttrSizedOperandSegments,
  AllTypesMatch<["target", "result"]>,
  Schedule_AffinityOp,
  Schedule_ScheduleableOp,
  Schedule_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{fills a subview of a stream resource with a value}];
  let description = [{
    Splats a value into a subview of the given stream resource and returns the
    resource with the update applied.

    Equivalent to a stream.tensor.splat + stream.tensor.update.
  }];

  let arguments = (ins
    Schedule_AnyScheduleResource:$target,
    TypeAttr:$target_encoding,
    Schedule_ShapeDynamicDims:$target_encoding_dims,
    Schedule_Size:$target_size,
    Variadic<Schedule_Dim>:$start_indices,
    Variadic<Schedule_Dim>:$lengths,
    Schedule_PrimitiveType:$value,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $value `,` $target `[` $start_indices `for` $lengths `]` `:`
    type($value)
    `->`
    $target_encoding (`` `{` $target_encoding_dims^ `}`)?
    `in`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getTargetEncodingDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getTargetEncodingDims(); }
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_TensorUpdateOp : Schedule_Op<"tensor.update", [
  AttrSizedOperandSegments,
  AllTypesMatch<["target", "result"]>,
  Schedule_AffinityOp,
  Schedule_ScheduleableOp,
  Schedule_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{updates a slice of a subview of a resource in-place}];
  let description = [{
    Copies a value into a resource based on tensor encodings. The returned value
    is the entire updated target value.
  }];

  let arguments = (ins
    Schedule_AnyScheduleResource:$target,
    TypeAttr:$target_encoding,
    Schedule_ShapeDynamicDims:$target_encoding_dims,
    Schedule_Size:$target_size,
    Variadic<Schedule_Dim>:$start_indices,
    Schedule_AnyScheduleResource:$update,
    TypeAttr:$update_encoding,
    Schedule_ShapeDynamicDims:$update_encoding_dims,
    Schedule_Size:$update_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $update `,` $target `[` $start_indices `]` `:`
    $update_encoding (`` `{` $update_encoding_dims^ `}`)?
    `in`
    type($update) `` `{` $update_size `}`
    `->`
    $target_encoding (`` `{` $target_encoding_dims^ `}`)?
    `in`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) {
      return idx == 0 ? getTargetEncodingDims() : getUpdateEncodingDims();
    }
    ValueRange getResultDynamicDims(unsigned idx) { return getTargetEncodingDims(); }
    Value getOperandSize(unsigned idx) {
      return idx == 0 ? getTargetSize() : getUpdateSize();
    }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Schedule_TensorLoadOp : Schedule_PureOp<"tensor.load", [
  AttrSizedOperandSegments,
  Schedule_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
]> {
  let summary = [{loads a value from a tensor element}];
  let description = [{
    Returns the element at the given location from within the tensor.
  }];

  let arguments = (ins
    Schedule_StagingResource:$source,
    TypeAttr:$source_encoding,
    Schedule_ShapeDynamicDims:$source_encoding_dims,
    Schedule_Size:$source_size,
    Variadic<Schedule_Dim>:$indices
  );
  let results = (outs
    AnyTypeOf<[Schedule_PrimitiveType, AnyVector]>:$result
  );

  let assemblyFormat = [{
    $source (`[` $indices^ `]`)? `:`
    $source_encoding (`` `{` $source_encoding_dims^ `}`)?
    `in`
    type($source) `` `{` $source_size `}`
    `->`
    type($result)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceEncodingDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return ValueRange{}; }
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_TensorStoreOp : Schedule_PureOp<"tensor.store", [
  AttrSizedOperandSegments,
  AllTypesMatch<["target", "result"]>,
  Schedule_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{stores a value into a tensor element}];
  let description = [{
    Returns a tensor with the element at the given index set to the given value.
  }];

  let arguments = (ins
    Schedule_StagingResource:$target,
    TypeAttr:$target_encoding,
    Schedule_ShapeDynamicDims:$target_encoding_dims,
    Schedule_Size:$target_size,
    Variadic<Schedule_Dim>:$indices,
    AnyTypeOf<[Schedule_PrimitiveType, AnyVector]>:$value
  );
  let results = (outs
    Schedule_StagingResource:$result
  );

  let assemblyFormat = [{
    $value `,`
    $target (`[` $indices^ `]`)? `:`
    type($value)
    `->`
    $target_encoding (`` `{` $target_encoding_dims^ `}`)?
    `in`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getTargetEncodingDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getTargetEncodingDims(); }
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_TensorTraceOp : Schedule_Op<"tensor.trace", []> {
  let summary = [{trace value(s) operation}];
  let description = [{
    Traces out to a runtime trace sink (console, log file, etc) the given
    tensors and titles them with the given key. The key is informational only
    and useful for titling/marking specific sets of tensors for easier
    searching.
  }];

  let arguments = (ins
    StrAttr:$key,
    Variadic<AnyTensor>:$operands
  );

  // TODO(benvanik): stream.tensor.trace assembly format (custom).
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Builtins
//===----------------------------------------------------------------------===//
// Builtins are added by frontends or during tensor encoding to handle ops or
// data types not supported by the lower-level stream ops (stream.async.* and
// stream.cmd.*).

def Schedule_BuiltinSplatI64Op : Schedule_Op<"builtin.splat.i64", [
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Schedule_BuiltinOp, [
    "mergeBuiltinModule",
    "convertBuiltinOp",
  ]>,
  Util_SizeAwareOp,
]> {
  let summary = [{splats a value into a resource}];
  let description = [{
    Generated builtin for emulating 64-bit splats.
    As with stream.async.splat it returns a new resource with the given
    value splatted out to fill the entire contents.
  }];

  let arguments = (ins
    I64:$value,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $value `:` type($value) `->` type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;
}

def Schedule_BuiltinFillI64Op : Schedule_Op<"builtin.fill.i64", [
  AllTypesMatch<["target", "result"]>,
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Schedule_BuiltinOp, [
    "mergeBuiltinModule",
    "convertBuiltinOp",
  ]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{fills a subview of a stream resource with a value}];
  let description = [{
    Generated builtin for emulating 64-bit fills.
    As with stream.async.fill it splats a value into a subview of the given
    stream resource and returns the resource with the update applied.
  }];

  let arguments = (ins
    Schedule_AnyScheduleResource:$target,
    Schedule_Size:$target_size,
    Schedule_Offset:$target_offset,
    Schedule_Offset:$target_end,
    Schedule_Size:$target_length,
    I64:$value,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $value `,`
    $target `[` $target_offset `to` $target_end `for` $target_length `]` `:`
    type($value) `->`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Resource transfer ops
//===----------------------------------------------------------------------===//

def Schedule_AsyncAllocaOp : Schedule_PureOp<"async.alloca", [
  DeclareOpInterfaceMethods<Schedule_AffinityOp, [
    "getAffinity",
    "setAffinity",
  ]>,
  Schedule_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Schedule_ScheduleableOp, [
    "isMetadata",
    "preferCloneToConsumers",
  ]>,
  Util_SizeAwareOp,
  MemoryEffects<[MemAlloc]>,
]> {
  let summary = [{allocates a transient value with undefined contents}];
  let description = [{
    Allocates a transient value (one that is short-lived and local to the
    current computation) with undefined contents. Consumers of the allocated
    result must assume nothing of the contents and use `discard` access.
  }];

  let arguments = (ins
    Schedule_Size:$storage_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_TransientResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    attr-dict `:` type($result) `{` $storage_size `}`
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getStorageSize(); }
  }];

  let hasCanonicalizer = 1;
}

def Schedule_AsyncConstantOp : Schedule_PureOp<"async.constant", [
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Schedule_ScheduleableOp, [
    "isMetadata",
  ]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{defines a constant resource}];
  let description = [{
    Returns a new resource with the given constant value.
  }];

  let arguments = (ins
    ElementsAttr:$value,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    `:`
    type($result) `` `{` $result_size `}`
    `=`
    $value
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_AsyncSplatOp : Schedule_Op<"async.splat", [
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Schedule_ScheduleableOp, [
    "preferCloneToConsumers",
  ]>,
  Util_SizeAwareOp,
]> {
  let summary = [{splats a value into a resource}];
  let description = [{
    Returns a new resource with the given primitive value splatted out to fill
    the entire contents.
  }];

  let arguments = (ins
    Schedule_FillPatternType:$value,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $value `:` type($value) `->` type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_AsyncCloneOp : Schedule_Op<"async.clone", [
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Schedule_ScheduleableOp, [
    "preferCloneToConsumers",
  ]>,
  Util_SizeAwareOp,
]> {
  let summary = [{clones the contents of a value}];
  let description = [{
    Clones the contents of a value at a snapshot in time. Future changes to the
    cloned value will not effect the result. Acts as a copy-on-write operation.
  }];

  let arguments = (ins
    Schedule_AnyScheduleResource:$source,
    Schedule_Size:$source_size,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `:`
    type($source) `` `{` $source_size `}` `->`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Schedule_AsyncSliceOp : Schedule_PureOp<"async.slice", [
  AllTypesMatch<["source", "result"]>,
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Schedule_ScheduleableOp, [
    "isMetadata",
  ]>,
  Util_SizeAwareOp,
]> {
  let summary = [{slices out a cloned subview of a value}];
  let description = [{
    Slices a subrange of a stream resource based on a byte range. Acts as a
    copy-on-write operation.
  }];

  let arguments = (ins
    Schedule_AnyScheduleResource:$source,
    Schedule_Size:$source_size,
    Schedule_Offset:$source_offset,
    Schedule_Offset:$source_end,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `[` $source_offset `to` $source_end `]` `:`
    type($source) `` `{` $source_size `}` `->`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Schedule_AsyncFillOp : Schedule_Op<"async.fill", [
  AllTypesMatch<["target", "result"]>,
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  Schedule_ScheduleableOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{fills a subview of a stream resource with a value}];
  let description = [{
    Splats a value into a subview of the given stream resource and returns the
    resource with the update applied.

    Equivalent to a stream.async.splat + stream.async.update.
  }];

  let arguments = (ins
    Schedule_AnyScheduleResource:$target,
    Schedule_Size:$target_size,
    Schedule_Offset:$target_offset,
    Schedule_Offset:$target_end,
    Schedule_Size:$target_length,
    Schedule_FillPatternType:$value,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $value `,`
    $target `[` $target_offset `to` $target_end `for` $target_length `]` `:`
    type($value) `->`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_AsyncUpdateOp : Schedule_Op<"async.update", [
  AllTypesMatch<["target", "result"]>,
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Schedule_ScheduleableOp, [
    "isMetadata",
  ]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{updates a slice of a subview of a resource in-place}];
  let description = [{
    Copies a value into a resource based on a byte range. The returned value
    is the entire updated target value. Updates can be turned into placement
    allocations and avoid copies.
  }];

  let arguments = (ins
    Schedule_AnyScheduleResource:$target,
    Schedule_Size:$target_size,
    Schedule_Offset:$target_offset,
    Schedule_Offset:$target_end,
    Schedule_AnyScheduleResource:$update,
    Schedule_Size:$update_size,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $update `,`
    $target `[` $target_offset `to` $target_end `]` `:`
    type($update) `` `{` $update_size `}` `->`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return idx == 0 ? getTargetSize() : getUpdateSize();
    }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Schedule_AsyncCopyOp : Schedule_Op<"async.copy", [
  AllTypesMatch<["target", "result"]>,
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  Schedule_ScheduleableOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{copies a subview of a stream resource to another}];
  let description = [{
    Copies a subview of a resource into a subview of another.
    As with memcpy this does not support overlapping updates into the same
    resource. Unlike `stream.async.update` copy sources cannot be allocated
    in-place.

    Equivalent to a stream.async.slice + stream.async.update.
  }];

  let arguments = (ins
    Schedule_AnyScheduleResource:$target,
    Schedule_Size:$target_size,
    Schedule_Offset:$target_offset,
    Schedule_Offset:$target_end,
    Schedule_AnyScheduleResource:$source,
    Schedule_Size:$source_size,
    Schedule_Offset:$source_offset,
    Schedule_Offset:$source_end,
    Schedule_Size:$length,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_AnyScheduleResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `[` $source_offset `to` $source_end `]` `,`
    $target `[` $target_offset `to` $target_end `]` `,`
    $length `:`
    type($source) `` `{` $source_size `}` `->`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return idx == 0 ? getTargetSize() : getSourceSize();
    }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_AsyncTransferOp : Schedule_Op<"async.transfer", [
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  Schedule_ScheduleableOp,
  Util_SizeAwareOp,
]> {
  let summary = [{transfers a resource from one location/state to another}];
  let description = [{
    Transfers a resource between different states (such as a `staging` lifetime
    to a `local` lifetime) or different affinities. This is roughly equivalent
    to a cast but may have special semantics when later lowered to one or more
    devices with discrete memory spaces or pools.
  }];

  let arguments = (ins
    AnyTypeOf<[
      Schedule_AnyScheduleResource,
      Schedule_StagingResource,
    ]>:$source,
    Schedule_Size:$source_size,
    Schedule_Size:$result_size,
    OptionalAttr<Schedule_AffinityAttr>:$source_affinity,
    OptionalAttr<Schedule_AffinityAttr>:$result_affinity
  );
  let results = (outs
    AnyTypeOf<[
      Schedule_AnyScheduleResource,
      Schedule_StagingResource,
    ]>:$result
  );

  let assemblyFormat = [{
    (`from` `(` $source_affinity^ `)`)?
    $source `:`
    type($source) `` `{` $source_size `}` `->`
    (`to` `(` $result_affinity^ `)`)?
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Schedule_AsyncLoadOp : Schedule_PureOp<"async.load", [
  Schedule_AsyncPhaseOp,
  Util_SizeAwareOp,
]> {
  let summary = [{loads a value from a resource}];
  let description = [{
    Returns the element at the given location from within the resource.
  }];

  let arguments = (ins
    Schedule_StagingResource:$source,
    Schedule_Size:$source_size,
    Schedule_Offset:$source_offset
  );
  let results = (outs
    AnyTypeOf<[Schedule_PrimitiveType, AnyVector]>:$result
  );

  let assemblyFormat = [{
    $source `[` $source_offset `]` `:`
    type($source) `` `{` $source_size `}`
    `->`
    type($result)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_AsyncStoreOp : Schedule_PureOp<"async.store", [
  AllTypesMatch<["target", "result"]>,
  Schedule_AsyncPhaseOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{stores a value into a resource}];
  let description = [{
    Returns a resource with the element at the given offset set to the given
    value.
  }];

  let arguments = (ins
    Schedule_StagingResource:$target,
    Schedule_Size:$target_size,
    Schedule_Offset:$target_offset,
    AnyTypeOf<[Schedule_PrimitiveType, AnyVector]>:$value
  );
  let results = (outs
    Schedule_StagingResource:$result
  );

  let assemblyFormat = [{
    $value `,`
    $target `[` $target_offset `]` `:`
    type($value)
    `->`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_AsyncDispatchOp : Schedule_Op<"async.dispatch", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  Schedule_ScheduleableOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedOperandsIndexAndLength",
  ]>,
]> {
  let summary = [{dispatches a parallelized grid of work}];
  let description = [{
    Calls the specified entry point function once for each element in the
    specified workgroup count. Each workgroup has access to the same operands
    and results and is able to load/store at will.
  }];

  let arguments = (ins
    Variadic<Index>:$workload,
    SymbolRefAttr:$entry_point,
    Variadic<AnyTypeOf<[
      Schedule_AnyScheduleResource,
      Schedule_PrimitiveType,
    ]>>:$resource_operands,
    Variadic<Schedule_Size>:$resource_operand_sizes,
    Variadic<Schedule_Size>:$result_sizes,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<Schedule_AnyScheduleResource>:$results
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $entry_point
    (`[` $workload^ `]`)? ``
    `(` $resource_operands `)` attr-dict `:`
    custom<ShapedFunctionType>(ref($resource_operands),
                               type($resource_operands), $resource_operand_sizes,
                               type($results), $result_sizes,
                               $tied_operands)
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), getResourceOperandSizes());
    }
    Value getResultSize(unsigned idx) {
      return findValueSizeInList(idx, getResults(), getResultSizes());
    }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Async control flow ops
//===----------------------------------------------------------------------===//

// TODO(benvanik): stream.async.if
// TODO(benvanik): stream.async.select
// TODO(benvanik): stream.async.for

def Schedule_AsyncExecuteOp : Schedule_Op<"async.execute", [
  AttrSizedOperandSegments,
  RecursiveSideEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getSuccessorEntryOperands",
  ]>,
  SingleBlockImplicitTerminator<"KLW::Schedule::YieldOp">,
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  Schedule_TimelineOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_ClosureOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResultsIndexAndLength",
  ]>,
]> {
  let summary = [{executes a dependency-aware sequence of streamable ops}];
  let description = [{
    Evaluates the operations within the region by dependency order while obeying
    ties when present. Nested ops execute serially in block order and nested
    `stream.async.concurrent` ops can be used to run multiple ops concurrently
    within the stream. All resource inputs must be captured explicitly. All
    results are only ready once all nested ops complete execution and the
    returned timepoint is reached. Zero or more timepoints may be provided to
    block execution until they are all reached; zero timepoints indicates that
    execution may begin immediately.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      Schedule_AnyScheduleResource,
      Schedule_StagingResource,
    ]>>:$resource_operands,
    Variadic<Schedule_Size>:$resource_operand_sizes,
    Variadic<Schedule_Size>:$result_sizes,
    Optional<Schedule_Timepoint>:$await_timepoint,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<AnyTypeOf<[
      Schedule_AnyScheduleResource,
      Schedule_StagingResource,
    ]>>:$results,
    Schedule_Timepoint:$result_timepoint
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    (`await` `(` $await_timepoint^ `)` `=` `` `>`)?
    `with` ``
    custom<ResourceRegion>($resource_operands,
                           type($resource_operands), $resource_operand_sizes,
                           type($results), $result_sizes,
                           $tied_operands, $body)
    `=` `` `>` type($result_timepoint)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "TypeRange":$resultTypes, "ValueRange":$resultSizes,
      "Value":$awaitTimepoint,
      "ValueRange":$resourceOperands, "ValueRange":$resourceOperandSizes,
      "ArrayRef<int64_t>":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), getResourceOperandSizes());
    }
    Value getResultSize(unsigned idx) {
      return findValueSizeInList(idx, getResults(), getResultSizes());
    }
    SmallVector<Value> getAwaitTimepoints() {
      if (getAwaitTimepoint()) return {getAwaitTimepoint()}; else return {};
    }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_AsyncConcurrentOp : Schedule_Op<"async.concurrent", [
  ParentOneOf<[
    "KLW::Schedule::AsyncExecuteOp",
    "KLW::Schedule::AsyncConcurrentOp",
  ]>,
  AttrSizedOperandSegments,
  RecursiveSideEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getSuccessorEntryOperands",
  ]>,
  SingleBlockImplicitTerminator<"KLW::Schedule::YieldOp">,
  Schedule_AffinityOp,
  Schedule_AsyncPhaseOp,
  Schedule_ScheduleableOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_ClosureOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface>,
]> {
  let summary = [{executes all ops concurrently}];
  let description = [{
    Represents a wave of work scheduled concurrently (each op executing at the
    same time). All resource inputs must be captured explicitly. All results are
    only ready once all nested ops complete execution.

    Waves can be nested to create a DAG. For example, take the following graph:
    ```
                      |
            v---------+---------v
    +-------|-------+   +-------|-------+
    |    v--+--v    |   |    v--+--v    |
    | +----+ +----+ |   | +----+ +----+ |
    | | %a | | %b | |   | | %c | | %d | |
    | +----+ +----+ |   | +----+ +----+ |
    |    +--v--+    |   |    +--v--+    |
    +-------|-------+   +-------|-------+
            +---------v---------+
                      |
    ```

    Represented with nested waves:
    ```mlir
      %0 = stream.async.concurrent with(%arg) -> ... {
        %1 = stream.async.concurrent with(%arg as %arg0) -> ... {
          %a = ...
          %b = ...
          stream.yield %a, %b
        }
        %2 = stream.async.concurrent with(%arg as %arg1) -> ... {
          %c = ...
          %d = ...
          stream.yield %c, %d
        }
        stream.yield %1, %2
      }
    ```
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      Schedule_AnyScheduleResource,
      Schedule_StagingResource,
    ]>>:$resource_operands,
    Variadic<Schedule_Size>:$resource_operand_sizes,
    Variadic<Schedule_Size>:$result_sizes,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<AnyTypeOf<[
      Schedule_AnyScheduleResource,
      Schedule_StagingResource,
    ]>>:$results
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    `with` ``
    custom<ResourceRegion>($resource_operands,
                           type($resource_operands), $resource_operand_sizes,
                           type($results), $result_sizes,
                           $tied_operands, $body)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "TypeRange":$resultTypes, "ValueRange":$resultSizes,
      "ValueRange":$resourceOperands, "ValueRange":$resourceOperandSizes,
      "ArrayRef<int64_t>":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), getResourceOperandSizes());
    }
    Value getResultSize(unsigned idx) {
      return findValueSizeInList(idx, getResults(), getResultSizes());
    }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Explicit command ops
//===----------------------------------------------------------------------===//

def Schedule_CmdFlushOp : Schedule_Op<"cmd.flush", [
  Schedule_CmdPhaseOp,
  Schedule_ScheduleableOp,
  Schedule_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{flushes a subview of a resource}];
  let description = [{
    Transfers a resource to an external target. The resource memory is made
    available to the target and can be made visible there using
    `stream.cmd.invalidate`.
  }];

  let arguments = (ins
    Schedule_AnyResource:$target,
    Schedule_Size:$target_size,
    Schedule_Offset:$target_offset,
    Schedule_Size:$target_length,
    OptionalAttr<Schedule_AffinityAttr>:$source_affinity
  );
  let results = (outs);

  let assemblyFormat = [{
    (`to` `(` $source_affinity^ `)`)?
    $target `[` $target_offset `for` $target_length `]` `:`
    type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_CmdInvalidateOp : Schedule_Op<"cmd.invalidate", [
  Schedule_CmdPhaseOp,
  Schedule_ScheduleableOp,
  Schedule_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{invalidates a subview of a resource}];
  let description = [{
    Transfers a resource from an external source into the current target. The
    resource memory is assumed to have been made available at the source via
    `stream.cmd.flush`.
  }];

  let arguments = (ins
    Schedule_AnyResource:$target,
    Schedule_Size:$target_size,
    Schedule_Offset:$target_offset,
    Schedule_Size:$target_length,
    OptionalAttr<Schedule_AffinityAttr>:$source_affinity
  );
  let results = (outs);

  let assemblyFormat = [{
    (`from` `(` $source_affinity^ `)`)?
    $target `[` $target_offset `for` $target_length `]` `:`
    type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_CmdDiscardOp : Schedule_Op<"cmd.discard", [
  Schedule_CmdPhaseOp,
  Schedule_ScheduleableOp,
  Schedule_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{discards a subview of a resource}];
  let description = [{
    Discards a subview of a resource, indicating that after this command the
    specified contents are no longer needed. This can be used to trim memory
    or invalidate caches.
  }];

  let arguments = (ins
    Schedule_AnyResource:$target,
    Schedule_Size:$target_size,
    Schedule_Offset:$target_offset,
    Schedule_Size:$target_length
  );
  let results = (outs);

  let assemblyFormat = [{
    $target `[` $target_offset `for` $target_length `]` `:`
    type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_CmdFillOp : Schedule_Op<"cmd.fill", [
  Schedule_CmdPhaseOp,
  Schedule_ScheduleableOp,
  Schedule_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{fills a subview of a stream resource with a value}];
  let description = [{
    Splats a value into a subview of the given stream resource and returns the
    resource with the update applied.
  }];

  let arguments = (ins
    Schedule_AnyScheduleResource:$target,
    Schedule_Size:$target_size,
    Schedule_Offset:$target_offset,
    Schedule_Size:$target_length,
    Schedule_FillPatternType:$value
  );
  let results = (outs);

  let assemblyFormat = [{
    $value `,`
    $target `[` $target_offset `for` $target_length `]` `:`
    type($value) `->`
    type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_CmdCopyOp : Schedule_Op<"cmd.copy", [
  Schedule_CmdPhaseOp,
  Schedule_ScheduleableOp,
  Schedule_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{copies a subview of a stream resource to another}];
  let description = [{
    Copies a subview of a resource into a subview of another.
    As with memcpy this does not support overlapping updates into the same
    resource.
  }];

  let arguments = (ins
    Schedule_AnyResource:$source,
    Schedule_Size:$source_size,
    Schedule_Offset:$source_offset,
    Schedule_AnyResource:$target,
    Schedule_Size:$target_size,
    Schedule_Offset:$target_offset,
    Schedule_Size:$length
  );
  let results = (outs);

  let assemblyFormat = [{
    $source `[` $source_offset `]` `,`
    $target `[` $target_offset `]` `,`
    $length `:`
    type($source) `` `{` $source_size `}` `->`
    type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return idx == 0 ? getSourceSize() : getTargetSize();
    }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_CmdDispatchOp : Schedule_Op<"cmd.dispatch", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Schedule_CmdPhaseOp,
  Schedule_ScheduleableOp,
  Schedule_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{dispatches a parallelized grid of work}];
  let description = [{
    Calls the specified entry point function once for each element in the
    specified workgroup count. Each workgroup has access to the same operands
    and results and is able to load/store at will.
  }];

  let arguments = (ins
    Variadic<Index>:$workload,
    SymbolRefAttr:$entry_point,
    Variadic<Schedule_PrimitiveType>:$uniform_operands,
    Variadic<Schedule_AnyScheduleResource>:$resources,
    Variadic<Schedule_Size>:$resource_sizes,
    Variadic<Schedule_Offset>:$resource_offsets,
    Variadic<Schedule_Size>:$resource_lengths,
    Schedule_ResourceAccessArrayAttr:$resource_accesses
  );
  let results = (outs);

  let assemblyFormat = [{
    $entry_point
    (`[` $workload^ `]`)? ``
    (`(` $uniform_operands^ `:` type($uniform_operands) `)`)? `{`
    custom<DispatchResources>($resources, type($resources), $resource_sizes,
                              $resource_offsets, $resource_lengths,
                              $resource_accesses)
    `\n` `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(
          idx - getODSOperandIndexAndLength(2).first,
          getResources(), getResourceSizes());
    }
    Value getResultSize(unsigned idx) { return {}; }

    // Builds a map of operand index to argument index.
    static SmallVector<unsigned> makeOperandToArgMap(mlir::func::FuncOp funcOp);
    // Builds a map of resource to argument index of the corresponding binding.
    static SmallVector<unsigned> makeResourceToArgMap(mlir::func::FuncOp funcOp);
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_CmdExecuteOp : Schedule_Op<"cmd.execute", [
  AttrSizedOperandSegments,
  RecursiveSideEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getSuccessorEntryOperands",
  ]>,
  SingleBlockImplicitTerminator<"KLW::Schedule::YieldOp">,
  Schedule_AffinityOp,
  Schedule_CmdPhaseOp,
  Schedule_TimelineOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_ClosureOpInterface>,
]> {
  let summary = [{executes a dependency-aware sequence of streamable ops}];
  let description = [{
    Evaluates the operations within the region by dependency order while obeying
    ties when present. Nested ops execute serially in block order and nested
    `stream.cmd.concurrent` ops can be used to run multiple ops concurrently
    within the stream. All resource inputs must be captured explicitly. All
    results are only ready once all nested ops complete execution and the
    returned timepoint is reached. Zero or more timepoints may be provided to
    block execution until they are all reached; zero timepoints indicates that
    execution may begin immediately.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      Schedule_AnyScheduleResource,
      Schedule_StagingResource,
    ]>>:$resource_operands,
    Variadic<Schedule_Size>:$resource_operand_sizes,
    Optional<Schedule_Timepoint>:$await_timepoint,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_Timepoint:$result_timepoint
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    (`await` `(` $await_timepoint^ `)` `=` `` `>`)?
    `with` ``
    custom<ExplicitResourceRegion>($resource_operands,
                                   type($resource_operands), $resource_operand_sizes,
                                   $body)
    `=` `` `>` type($result_timepoint)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Value":$awaitTimepoint,
      "ValueRange":$resourceOperands, "ValueRange":$resourceOperandSizes,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), getResourceOperandSizes());
    }
    Value getResultSize(unsigned idx) {
      return {};
    }
    SmallVector<Value> getAwaitTimepoints() {
      if (getAwaitTimepoint()) return {getAwaitTimepoint()}; else return {};
    }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

// TODO(benvanik): stream.cmd.serial for serialized execution.
def Schedule_CmdSerialOp : Schedule_Op<"cmd.serial", [
  ParentOneOf<[
    "KLW::Schedule::CmdExecuteOp",
    "KLW::Schedule::CmdSerialOp",
    "KLW::Schedule::CmdConcurrentOp",
  ]>,
  RecursiveSideEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface>,
  SingleBlockImplicitTerminator<"KLW::Schedule::YieldOp">,
  Schedule_CmdPhaseOp,
  Schedule_ScheduleableOp,
]> {
  let summary = [{executes all ops serially (in-order)}];
  let description = [{
    Represents a sequence of work scheduled serially (each op executing one
    after the other).

    Regions can be nested to create a DAG. For example, take the following graph:
    ```
                      |
            v---------+-----v
    +-------|-------+   +---|----+
    |    v--+--v    |   |   v    |
    | +----+ +----+ |   | +----+ |
    | | @a | | @b | |   | | @c | |
    | +----+ +----+ |   | +----+ |
    |    |     |    |   |   |    |
    |    |     |    |   | +-v--+ |
    |    |     |    |   | | @d | |
    |    |     |    |   | +----+ |
    |    +--v--+    |   |   |    |
    +-------|-------+   +---|----+
            +---------v-----+
                      |
    ```

    Represented with nested regions:
    ```mlir
      stream.cmd.concurrent {
        stream.cmd.concurrent {
          stream.cmd.dispatch @a
          stream.cmd.dispatch @b
        }
        stream.cmd.serial {
          stream.cmd.dispatch @c
          stream.cmd.dispatch @d
        }
      }
    ```
  }];

  let arguments = (ins);
  let results = (outs);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $body
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Schedule_CmdConcurrentOp : Schedule_Op<"cmd.concurrent", [
  ParentOneOf<[
    "KLW::Schedule::CmdExecuteOp",
    "KLW::Schedule::CmdSerialOp",
    "KLW::Schedule::CmdConcurrentOp",
  ]>,
  RecursiveSideEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface>,
  SingleBlockImplicitTerminator<"KLW::Schedule::YieldOp">,
  Schedule_CmdPhaseOp,
  Schedule_ScheduleableOp,
]> {
  let summary = [{executes all ops concurrently}];
  let description = [{
    Represents a wave of work scheduled concurrently (each op executing at the
    same time).

    Waves can be nested to create a DAG. For example, take the following graph:
    ```
                      |
            v---------+---------v
    +-------|-------+   +-------|-------+
    |    v--+--v    |   |    v--+--v    |
    | +----+ +----+ |   | +----+ +----+ |
    | | @a | | @b | |   | | @c | | @d | |
    | +----+ +----+ |   | +----+ +----+ |
    |    +--v--+    |   |    +--v--+    |
    +-------|-------+   +-------|-------+
            +---------v---------+
                      |
    ```

    Represented with nested waves:
    ```mlir
      stream.cmd.concurrent {
        stream.cmd.concurrent {
          stream.cmd.dispatch @a
          stream.cmd.dispatch @b
        }
        stream.cmd.concurrent {
          stream.cmd.dispatch @c
          stream.cmd.dispatch @d
        }
      }
    ```
  }];

  let arguments = (ins);
  let results = (outs);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $body
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Synchronization ops
//===----------------------------------------------------------------------===//

def Schedule_TimepointImmediateOp : Schedule_PureOp<"timepoint.immediate", [
  ConstantLike,
  Schedule_TimelineOp,
]> {
  let summary = [{results an immediately-available timepoint}];
  let description = [{
    Timepoints indicate a point in the execution timeline and this op can be
    used to get a placeholder representing the start of the timeline. Any waits
    on the returned timepoint will resolve immediately. This generally folds
    away but can be useful if needing to initialize globals or branch args.
  }];

  let arguments = (ins);
  let results = (outs
    Schedule_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    attr-dict
    `=` `` `>` type($result_timepoint)
  }];

  let extraClassDeclaration = [{
    SmallVector<Value> getAwaitTimepoints() { return {}; }
  }];

  let hasFolder = 1;
}

def Schedule_TimepointImportOp : Schedule_PureOp<"timepoint.import", [
  Schedule_AffinityOp,
]> {
  let summary = [{imports a timepoint from an external dialect type}];
  let description = [{
    Defines a conversion from an external dialect type such as `hal.semaphore`
    that is resolved during lowering into the stream dialect. This can be used
    to interoperate between levels of the stack that require specifying stream
    types and those that prior to lowering do not handle them.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Schedule_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $operands `:` `(` type($operands) `)`
    `=` `` `>`
    type($result_timepoint)
    attr-dict-with-keyword
  }];
}

def Schedule_TimepointExportOp : Schedule_PureOp<"timepoint.export", [
  Schedule_AffinityOp,
]> {
  let summary = [{exports a timepoint to an external dialect type}];
  let description = [{
    Defines a conversion to an external dialect type such as `hal.semaphore`
    that is resolved during lowering into the stream dialect. This can be used
    to interoperate between levels of the stack that require specifying stream
    types and those that prior to lowering do not handle them.
  }];

  let arguments = (ins
    Schedule_Timepoint:$await_timepoint,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $await_timepoint
    `=` `` `>`
    `(` type($results) `)`
    attr-dict-with-keyword
  }];

  let hasFolder = 1;
}

def Schedule_TimepointJoinOp : Schedule_PureOp<"timepoint.join", [
  Schedule_TimelineOp,
]> {
  let summary = [{joins one or more timepoints into the max of all of them}];
  let description = [{
    Returns a timepoint that indicates that all of the input timepoints have
    been reached.
  }];

  let arguments = (ins
    Variadic<Schedule_Timepoint>:$await_timepoints
  );
  let results = (outs
    Schedule_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    `max` `(` $await_timepoints `)` `=` `` `>` type($result_timepoint)
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Schedule_TimepointAwaitOp : Schedule_PureOp<"timepoint.await", [
  AttrSizedOperandSegments,
  Schedule_AffinityOp,
  Schedule_TimelineOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{awaits a timepoint before returning a set of resources}];
  let description = [{
    After asynchronous execution scheduling resources may exist in different
    states at different points in the execution timeline. This op enables
    resolving the version of a resource after a particular point in the
    timeline. As timepoints transitively chain the timepoint must only cover the
    resource availability but not be limited to its original production
    timepoint.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      Schedule_AnyScheduleResource,
      Schedule_StagingResource,
    ]>>:$resource_operands,
    Variadic<Schedule_Size>:$resource_operand_sizes,
    Schedule_Timepoint:$await_timepoint,
    OptionalAttr<Schedule_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<AnyTypeOf<[
      Schedule_AnyScheduleResource,
      Schedule_StagingResource,
    ]>>:$results
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $await_timepoint `=` `` `>`
    $resource_operands `:`
    custom<SizeAwareTypeList>(type($resource_operands),
                              type($results), $resource_operand_sizes)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "ValueRange":$resourceOperands, "ValueRange":$resourceOperandSizes,
      "Value":$await_timepoint,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return getResourceOperandSizes()[idx];
    }
    Value getResultSize(unsigned idx) {
      return getResourceOperandSizes()[idx];
    }
    SmallVector<Value> getAwaitTimepoints() {
      if (getAwaitTimepoint()) return {getAwaitTimepoint()}; else return {};
    }
    Value getResultTimepoint() { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Executables
//===----------------------------------------------------------------------===//

def Schedule_ExecutableOp : Schedule_Op<"executable", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"KLW::Schedule::ExecutableEndOp">,
  NativeOpTrait<"SymbolTable">,
  Symbol,
]> {
  let summary = [{generic executable module}];
  let description = [{
    An executable module containing one or more public functions. The contents
    of the functions are safe to dispatch and can be lowered further to
    target-specific backend IR representations.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    attr-dict-with-keyword
    regions
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name)>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return getBody().front(); }
    ::mlir::ModuleOp getInnerModule() {
      return *getBlock().getOps<::mlir::ModuleOp>().begin();
    }
  }];

  let hasVerifier = 1;
}

def Schedule_ExecutableEndOp : Schedule_Op<"executable.end", [
  HasParent<"KLW::Schedule::ExecutableOp">,
  Terminator,
]> {
  let summary = [{terminator pseudo-op for the executable op}];
  let assemblyFormat = "attr-dict";
}

def Schedule_ExecutableExportOp : Schedule_Op<"executable.export", [
  HasParent<"KLW::Schedule::ExecutableOp">,
  Symbol,
  IsolatedFromAbove,
]> {
  let summary = [{defines an executable entry point for dispatch operations}];
  let description = [{
    Specifies an exported function with an externally-visible alias. Multiple
    exports can reference the same internal function.

    Each entry point can have a unique workgroup count calculation region.
    This region takes the workload parameters passed to each flow.dispatch and
    produces an XYZ workgroup count for the 3D grid dispatch.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$function_ref
  );

  let regions = (region AnyRegion:$workgroup_count);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    custom<SymbolAlias>($sym_name, $function_ref)
    custom<WorkgroupCountRegion>($workgroup_count)
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "StringRef":$sym_name,
      "FlatSymbolRefAttr":$function_ref
    )>,
  ];

  let extraClassDeclaration = [{
    ::mlir::func::FuncOp lookupFunctionRef();
  }];

  let hasVerifier = 1;
}

def Schedule_BindingSubspanOp : Schedule_PureOp<"binding.subspan", [
    Util_ShapeAwareOp,
  ]> {
  let summary = [{returns an alias to a subspan of interface binding data}];
  let description = [{
    Returns a subview to a tensor or memref-like type from a binding. The same
    binding may have multiple subviews at different byte offsets.
  }];

  let arguments = (ins
    Schedule_AnyBinding:$binding,
    Schedule_Offset:$byte_offset,
    Schedule_ShapeDynamicDims:$dynamic_dims
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $binding `` `[` $byte_offset `]`
    attr-dict `:` type($binding) `->` type($result) (`{` $dynamic_dims^ `}`)?
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getDynamicDims(); }
  }];
}

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

def Schedule_ReturnOp : Schedule_Op<"return", [
  ParentOneOf<[
    "KLW::Schedule::ExecutableExportOp",
  ]>,
  NoSideEffect,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface, [
    "getMutableSuccessorOperands",
  ]>,
  ReturnLike,
  Terminator,
]> {
  let summary = [{returns results from a region}];
  let description = [{
    The values returned are copied by-value.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = [{
    attr-dict
    $operands `:` type($operands)
  }];
}

def Schedule_YieldOp : Schedule_Op<"yield", [
  ParentOneOf<[
    "KLW::Schedule::AsyncExecuteOp",
    "KLW::Schedule::AsyncConcurrentOp",
    "KLW::Schedule::CmdExecuteOp",
    "KLW::Schedule::CmdSerialOp",
    "KLW::Schedule::CmdConcurrentOp",
  ]>,
  NoSideEffect,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface, [
    "getMutableSuccessorOperands",
  ]>,
  ReturnLike,
  Terminator,
  SameVariadicOperandSize,
  Util_SizeAwareOp,
]> {
  let summary = [{yields stream values from an execution region}];
  let description = [{
    The values returned represent the asynchronous value at the point in time
    the SSA value is defined (or tied).
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      Schedule_AnyScheduleResource,
      Schedule_StagingResource,
    ]>>:$resource_operands,
    Variadic<Index>:$resource_operand_sizes
  );

  let assemblyFormat = [{
    attr-dict
    ($resource_operands^ `:`
        custom<SizeAwareTypeList>(type($resource_operands),
                                  $resource_operand_sizes))?
  }];

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, ValueRange{}, ValueRange{});
    }]>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), getResourceOperandSizes());
    }
    Value getResultSize(unsigned idx) { return {}; }
  }];
}

#endif  // KLW_DIALECT_SCHEDULE_OPS
