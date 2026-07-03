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

from typing import TypeVar, List, Union
from functools import wraps
import enum

from triton._C.libtriton import ir, ascend_ir
import triton.language.core as tl
import triton.extension.buffer.language as bl
from . import semantic as semantic

from triton.language.core import (
    _constexpr_to_value
)


__all__ = [
    "sub_vec_id",
    "sync_block_set",
    "sync_block_wait",
    "address_space",
    "copy_from_ub_to_l1",
    "fixpipe",
    "PIPE",
]

T = TypeVar("T")

TRITON_BUILTIN = "__triton_builtin__"
ASCEND_BUILTIN = "__ascend_builtin__"


def builtin(fn: T) -> T:
    """Mark a function as a buffer language builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    # also set triton_builtin to true so that CodeGenerator will recognize this function
    setattr(wrapper, TRITON_BUILTIN, True)
    setattr(wrapper, ASCEND_BUILTIN, True)

    return wrapper


def is_builtin(fn) -> bool:
    """Is this a registered ascend language builtin function?"""
    return getattr(fn, ASCEND_BUILTIN, False)

class PIPE(enum.Enum):
    PIPE_S = ascend_ir.PIPE.PIPE_S
    PIPE_V = ascend_ir.PIPE.PIPE_V
    PIPE_M = ascend_ir.PIPE.PIPE_M
    PIPE_MTE1 = ascend_ir.PIPE.PIPE_MTE1
    PIPE_MTE2 = ascend_ir.PIPE.PIPE_MTE2
    PIPE_MTE3 = ascend_ir.PIPE.PIPE_MTE3
    PIPE_ALL = ascend_ir.PIPE.PIPE_ALL
    PIPE_FIX = ascend_ir.PIPE.PIPE_FIX

@builtin
def sub_vec_id(_builder=None) -> tl.tensor:
    """
    Get the Vector Core index on the AI Core.
    """
    return semantic.sub_vec_id(_builder)


@builtin
def sync_block_set(sender, receiver, event_id, sender_pipe: PIPE, receiver_pipe: PIPE, _builder=None):
    sender = _constexpr_to_value(sender)
    receiver = _constexpr_to_value(receiver)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(sender, str) and (sender == "cube" or sender == "vector"), f"ERROR: sender = {sender}, only supports cube/vector"
    assert isinstance(receiver, str) and (receiver == "cube" or receiver == "vector"), f"ERROR: receiver = {receiver}, only supports cube/vector"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f'Unexpected pair: {sender} -> {receiver}, only supports cube -> vector or vector -> cube')
    if not isinstance(sender_pipe, PIPE) or not isinstance(receiver_pipe, PIPE):
        raise TypeError("sender_pipe and receiver_pipe must be instances of PIPE enum")
    _builder.sync_block_set(sender, receiver, event_id, sender_pipe.value, receiver_pipe.value)


@builtin
def sync_block_wait(sender, receiver, event_id, sender_pipe: PIPE, receiver_pipe: PIPE, _builder=None):
    sender = _constexpr_to_value(sender)
    receiver = _constexpr_to_value(receiver)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(sender, str) and (sender == "cube" or sender == "vector"), f"ERROR: sender = {sender}, only supports cube/vector"
    assert isinstance(receiver, str) and (receiver == "cube" or receiver == "vector"), f"ERROR: receiver = {receiver}, only supports cube/vector"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f'Unexpected pair: {sender} -> {receiver}, only supports cube -> vector or vector -> cube')
    if not isinstance(sender_pipe, PIPE) or not isinstance(receiver_pipe, PIPE):
        raise TypeError("sender_pipe and receiver_pipe must be instances of PIPE enum")
    _builder.sync_block_wait(sender, receiver, event_id, sender_pipe.value, receiver_pipe.value)


class ascend_address_space(bl.address_space):
    def __init__(self, address_space_value: ascend_ir.AddressSpace) -> None:
        super().__init__()
        self.real_address_space = address_space_value

    def to_ir(self, fn, builder: ascend_ir.ascendnpu_ir_builder) -> ir.attribute:
        return fn(self.real_address_space, builder)


class ascend_address_space_group:

    def __init__(self):
        for k, v in {
            k: v
            for k, v in ascend_ir.AddressSpace.__dict__.items()
            if isinstance(v, ascend_ir.AddressSpace)
        }.items():
            setattr(self, k, ascend_address_space(v))


address_space = ascend_address_space_group()


@builtin
def copy_from_ub_to_l1(
    src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], _builder: None
) -> None:
    """
    Copies data from the Unified Buffer (UB) to the L1 Buffer.

    :param src: The source data located in the Unified Buffer.
    :type src: tl.tensor | bl.buffer
    :param dst: The destination buffer located in L1 memory.
    :type dst: tl.tensor | bl.buffer
    """
    return semantic.copy_from_ub_to_l1(src, dst, _builder)


@builtin
def fixpipe(src: tl.tensor, dst: bl.buffer, dma_mode: str = "nz2nd", _builder=None) -> None:
    """
    Directly store a tensor on L0C to a local buffer via fixpipe.
    Fixpipe is pipeline that performing data movement from L0C to other memory hierarchies.
    Currently support:
        - L0C to UB (for Ascend910_95 sereies)

    :param src: the source tensor, Must be located in the l0C memory region.
    :type src: tl.tensor
    :param dst: The destination buffer, Must be located in the UB memory region.
    :type dst: bl.buffer
    :param dma_mode: DMA transfer mode, "nz2nd" enables NZ to ND layout transformation
    :type dma_mode: str
    """
    if isinstance(src, tl.tensor) == False:
        raise TypeError("src is not of tensor type")
    elif isinstance(dst, bl.buffer) == False:
        raise TypeError("dst is not of buffer type")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst need to have the same type")
    if src.shape != dst.shape:
        raise TypeError("src and dst need to have the same shape")
    if dst.space != address_space.UB:
        raise TypeError("dst must be located in the UB memory region")
    if dma_mode not in ["nz2nd", "default"]:
        raise ValueError(f"dma_mode must be 'nz2nd' or 'default', got '{dma_mode}'")
    return semantic.fixpipe(src, dst, dma_mode, _builder)
