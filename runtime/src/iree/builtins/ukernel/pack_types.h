// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_PACK_TYPES_H_
#define IREE_BUILTINS_UKERNEL_PACK_TYPES_H_

#include <assert.h>

#include "iree/builtins/ukernel/common.h"

typedef enum iree_uk_pack_type_t {
  iree_uk_pack_type_f32f32 = IREE_UK_PACK_2_TYPES_LITERAL(FLOAT_32, FLOAT_32),
  iree_uk_pack_type_i8i8 = IREE_UK_PACK_2_TYPES_LITERAL(INT_8, INT_8),
  iree_uk_pack_type_i32i32 = IREE_UK_PACK_2_TYPES_LITERAL(INT_32, INT_32),
} iree_uk_pack_type_t;

static inline iree_uk_type_t iree_uk_pack_in_type(iree_uk_pack_type_t type) {
  return IREE_UK_UNPACK_TYPE(0, type);
}

static inline iree_uk_type_t iree_uk_pack_out_type(iree_uk_pack_type_t type) {
  return IREE_UK_UNPACK_TYPE(1, type);
}

// Parameters for a pack operation.
typedef struct iree_uk_pack_params_t {
  iree_uk_pack_type_t type;
  const void* in_buffer;
  void* out_buffer;
  iree_uk_ssize_t in_stride0;
  iree_uk_ssize_t out_stride0;
  iree_uk_ssize_t in_size0;
  iree_uk_ssize_t in_size1;
  iree_uk_ssize_t out_size0;
  iree_uk_ssize_t out_size1;
  iree_uk_ssize_t out_size2;
  iree_uk_ssize_t out_size3;
  const void* padding_value;
  iree_uk_uint32_t flags;
} iree_uk_pack_params_t;

#endif  // IREE_BUILTINS_UKERNEL_PACK_TYPES_H_
