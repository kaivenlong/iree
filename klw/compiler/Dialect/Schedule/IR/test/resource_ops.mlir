// RUN: klw-opt --split-input-file %s | klw-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @resourceAlloc
func.func @resourceAlloc(%arg0: index, %arg1: index) -> (!schedule.resource<*>, !schedule.resource<*>) {
  // CHECK: = schedule.resource.alloc uninitialized : !schedule.resource<*>{%arg0}, !schedule.resource<*>{%arg1}
  %0:2 = schedule.resource.alloc uninitialized : !schedule.resource<*>{%arg0}, !schedule.resource<*>{%arg1}
  return %0#0, %0#1 : !schedule.resource<*>, !schedule.resource<*>
}

// -----

// CHECK-LABEL: @resourceAlloca
func.func @resourceAlloca(%arg0: index, %await_timepoint: !schedule.timepoint) -> (!schedule.resource<staging>, !schedule.timepoint, !schedule.resource<staging>, !schedule.timepoint) {
  // CHECK: = schedule.resource.alloca uninitialized : !schedule.resource<staging>{%arg0} => !schedule.timepoint
  %0:2 = schedule.resource.alloca uninitialized : !schedule.resource<staging>{%arg0} => !schedule.timepoint
  // CHECK: = schedule.resource.alloca uninitialized await(%arg1) => !schedule.resource<staging>{%arg0} => !schedule.timepoint
  %1:2 = schedule.resource.alloca uninitialized await(%await_timepoint) => !schedule.resource<staging>{%arg0} => !schedule.timepoint
  return %0#0, %0#1, %1#0, %1#1 : !schedule.resource<staging>, !schedule.timepoint, !schedule.resource<staging>, !schedule.timepoint
}

// -----

// CHECK-LABEL: @resourceDealloca
func.func @resourceDealloca(%arg0: index, %arg1: !schedule.resource<staging>, %arg2: !schedule.timepoint) {
  // CHECK: = schedule.resource.dealloca %arg1 : !schedule.resource<staging>{%arg0} => !schedule.timepoint
  schedule.resource.dealloca %arg1 : !schedule.resource<staging>{%arg0} => !schedule.timepoint
  // CHECK: = schedule.resource.dealloca await(%arg2) => %arg1 : !schedule.resource<staging>{%arg0} => !schedule.timepoint
  schedule.resource.dealloca await(%arg2) => %arg1 : !schedule.resource<staging>{%arg0} => !schedule.timepoint
  return
}

// -----

// CHECK-LABEL: @resourceSize
func.func @resourceSize(%arg0: !schedule.resource<*>) -> index {
  // CHECK: = schedule.resource.size %arg0 : !schedule.resource<*>
  %0 = schedule.resource.size %arg0 : !schedule.resource<*>
  return %0 : index
}

// -----

// CHECK-LABEL: @resourceMap
func.func @resourceMap(%arg0: !util.buffer) -> !schedule.resource<staging> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = schedule.resource.map %arg0[%c0] : !util.buffer -> !schedule.resource<staging>{%c128}
  %0 = schedule.resource.map %arg0[%c0] : !util.buffer -> !schedule.resource<staging>{%c128}
  return %0 : !schedule.resource<staging>
}

// -----

// CHECK-LABEL: @resourceTryMap
func.func @resourceTryMap(%arg0: !util.buffer) -> (i1, !schedule.resource<constant>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = schedule.resource.try_map %arg0[%c0] : !util.buffer -> i1, !schedule.resource<constant>{%c128}
  %0:2 = schedule.resource.try_map %arg0[%c0] : !util.buffer -> i1, !schedule.resource<constant>{%c128}
  return %0#0, %0#1 : i1, !schedule.resource<constant>
}

// -----

// CHECK-LABEL: @resourceLoad
func.func @resourceLoad(%arg0: !schedule.resource<staging>, %arg1: index) -> i32 {
  %c0 = arith.constant 0 : index
  // CHECK: = schedule.resource.load %arg0[%c0] : !schedule.resource<staging>{%arg1} -> i32
  %0 = schedule.resource.load %arg0[%c0] : !schedule.resource<staging>{%arg1} -> i32
  return %0 : i32
}

// -----

// CHECK-LABEL: @resourceStore
func.func @resourceStore(%arg0: !schedule.resource<staging>, %arg1: index) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: schedule.resource.store %c123_i32, %arg0[%c0] : i32 -> !schedule.resource<staging>{%arg1}
  schedule.resource.store %c123_i32, %arg0[%c0] : i32 -> !schedule.resource<staging>{%arg1}
  return
}

// -----

// CHECK-LABEL: @resourcePack
func.func @resourcePack(%arg0: index, %arg1: index) -> (index, index, index) {
  %c128 = arith.constant 128 : index
  //      CHECK: schedule.resource.pack offset(%c128) slices({
  // CHECK-NEXT:   [0, 9] = %arg0,
  // CHECK-NEXT:   [3, 8] = %arg1
  // CHECK-NEXT:  })
  %0:3 = schedule.resource.pack offset(%c128) slices({
    [0, 9] = %arg0,
    [3, 8] = %arg1,
  }) : index
  return %0#0, %0#1, %0#2 : index, index, index
}

// -----

// CHECK-LABEL: @resourceConstants
func.func @resourceConstants() -> (!schedule.resource<constant>, !schedule.resource<constant>, !schedule.timepoint) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  //      CHECK: = schedule.resource.constants :
  // CHECK-NEXT:   !schedule.resource<constant>{%c4} = dense<100> : tensor<1xi32>,
  // CHECK-NEXT:   !schedule.resource<constant>{%c8} = dense<[101, 102]> : tensor<2xi32>
  // CHECK-NEXT:   => !schedule.timepoint
  %0:3 = schedule.resource.constants :
    !schedule.resource<constant>{%c4} = dense<100> : tensor<1xi32>,
    !schedule.resource<constant>{%c8} = dense<[101, 102]> : tensor<2xi32>
    => !schedule.timepoint
  return %0#0, %0#1, %0#2 : !schedule.resource<constant>, !schedule.resource<constant>, !schedule.timepoint
}

// -----

// CHECK-LABEL: @resourceSubview
func.func @resourceSubview(%arg0: !schedule.resource<*>, %arg1: index) -> !schedule.resource<*> {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK: = schedule.resource.subview %arg0[%c128] : !schedule.resource<*>{%arg1} -> !schedule.resource<*>{%c256}
  %0 = schedule.resource.subview %arg0[%c128] : !schedule.resource<*>{%arg1} -> !schedule.resource<*>{%c256}
  return %0 : !schedule.resource<*>
}
