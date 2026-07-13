# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import (
    TypeVar, List
)

from triton._C.libtriton import ir, buffer_ir
import triton.language.core as tl

from . import core as bl


T = TypeVar('T')


def allocate_local_buffer(
    type: tl.dtype,
    shape: List[tl.constexpr],
    address_space: bl.address_space,
    builder: buffer_ir.buffer_builder
) -> bl.buffer:
    shape = tl._unwrap_shape(shape)
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape must be list/tuple")
    type = tl._constexpr_to_value(type)
    address_space = tl._constexpr_to_value(address_space)

    buffer_ty = bl.buffer_type(element_ty=type, shape=shape, space=address_space)
    buffer_ty_ir = buffer_ty.to_ir(builder)

    return bl.buffer(builder.allocate_local_buffer(buffer_ty_ir), buffer_ty)


def to_buffer(
    tensor: tl.tensor,
    address_space: bl.address_space,
    builder: ir.builder,
) -> bl.buffer:
    if not isinstance(tensor.shape, (tuple, list)) or not tensor.shape:
        raise TypeError("scalar type cannot be converted to buffer")

    fn, _builder = builder.codegen_fn_builder_pairs.get("create_address_space")
    if fn is None or _builder is None:
        raise ValueError("Builder must have codegen_fn_builder_pairs with create_address_space")

    addr_space_attr = address_space.to_ir(fn, _builder) if address_space else builder.get_null_attr()

    handle = builder.to_buffer(tensor.handle, addr_space_attr)
    buffer_ty = bl.buffer_type(element_ty=tensor.dtype, shape=tensor.shape, space=address_space)

    return bl.buffer(handle, buffer_ty)



def to_tensor(
    memref: bl.buffer,
    writable: bool,
    builder: ir.builder
) -> tl.tensor:
    if not isinstance(memref, bl.buffer):
        raise TypeError("memref must be bl.buffer")
    tensor_type = tl.block_type(memref.dtype, memref.shape)
    return tl.tensor(builder.to_tensor(memref.handle, writable), tensor_type)

def subview(
    src: bl.buffer,
    offsets: List[tl.tensor],
    sizes: List[tl.constexpr],
    strides: List[tl.constexpr],
    builder: ir.builder
) -> bl.buffer:
    new_offsets = [o.handle for o in offsets]
    sizes_int = tl._unwrap_shape(sizes)
    strides_int = tl._unwrap_shape(strides)

    result_handle = builder.subview(src.handle, new_offsets, sizes_int, strides_int)
    buffer_ty = bl.buffer_type(element_ty=src.dtype, shape=sizes_int, space=src.space)
    return bl.buffer(result_handle, buffer_ty)
