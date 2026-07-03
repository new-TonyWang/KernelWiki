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

from triton._C.libtriton import ir
import triton.language.core as tl

from . import core as bl


T = TypeVar('T')


def allocate_local_buffer(
    type: tl.dtype,
    shape: List[tl.constexpr],
    address_space: bl.address_space,
    builder: ir.builder
) -> bl.buffer:
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape must be list/tuple")
    shape = tl._unwrap_shape(shape)
    element_ty = type.to_ir(builder)
    address_space = address_space.to_ir(
        builder) if address_space else builder.get_null_attr()
    return bl.buffer(builder.allocate_local_buffer(element_ty, shape, address_space),
                     dtype=type, shape=shape, space=address_space)
