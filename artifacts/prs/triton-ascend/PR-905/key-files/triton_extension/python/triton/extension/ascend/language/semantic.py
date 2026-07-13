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

from typing import Union
from triton._C.libtriton import ir, ascend_ir
import triton.language.core as tl
import triton.extension.buffer.language.core as bl
from . import core as al

__all__ = ["create_address_space"]


def sub_vec_id(builder: ascend_ir.ascendnpu_ir_builder) -> tl.tensor:
    return tl.tensor(builder.create_get_sub_vec_id(), tl.int64)


def fixpipe(src: tl.tensor, dst: bl.buffer, dma_mode: str, builder: ir.builder) -> None:
    enable_nz2nd = (dma_mode == "nz2nd")
    builder.create_fixpipe(src.handle, dst.handle, enable_nz2nd)

def create_address_space(
    address_space: ascend_ir.AddressSpace, builder: ascend_ir.ascendnpu_ir_builder
) -> ir.attribute:
    return builder.get_target_attribute(address_space)


def copy_from_ub_to_l1(
    src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], builder
):
    if isinstance(src, tl.tensor) or isinstance(dst, tl.tensor):
        raise TypeError("tensor not support yet")
    if src.shape != dst.shape:
        raise TypeError("src and dst must have same shape")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst need to have the same type")
    if isinstance(src, bl.buffer) and isinstance(dst, bl.buffer):
        if src.space != al.address_space.UB:
            raise TypeError("src's AddressSpace must be UB")
        if dst.space != al.address_space.L1:
            raise TypeError("dst's AddressSpace must be L1")
        builder.create_copy_buffer(src.handle, dst.handle)
    else:
        raise TypeError("src and dst must be tl.tensor or bl.buffer")
