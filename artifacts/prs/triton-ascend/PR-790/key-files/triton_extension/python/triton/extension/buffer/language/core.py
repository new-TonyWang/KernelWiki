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

from typing import TypeVar, List
from functools import wraps

from triton._C.libtriton import ir
import triton.language.core as tl
from triton.language import semantic as real_semantic

__all__ = [
    "address_space",
    "allocate_local_buffer",
    "to_buffer",
    "to_tensor",
    "buffer",
    "subview"
]

T = TypeVar("T")

TRITON_BUILTIN = "__triton_builtin__"
BUFFER_BUILTIN = "__buffer_builtin__"


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
    setattr(wrapper, BUFFER_BUILTIN, True)

    return wrapper


def is_builtin(fn) -> bool:
    """Is this a registered buffer language builtin function?"""
    return getattr(fn, BUFFER_BUILTIN, False)


class address_space:
    """Represents a buffer's address space.

    The :code:`address_space` of a buffer is a target-specific attribute.
    """

    def to_ir(self, builder: ir.builder) -> ir.type:
        raise NotImplementedError(
            "Abstract address_space cannot be converted to ir"
        )

# -----------------------
# buffer
# -----------------------


class buffer(tl._value):
    """Represents a region of memory.

    :code:`buffer` is the fundamental data structure for Triton programs using
    the buffer language extension. Most functions in
    :py:mod:`triton.extension.buffer.language` operate on and return buffers.

    Most of the named member functions here are duplicates of the free functions
    in :code:`triton.language`.  For example, :code:`triton.language.sqrt(x)` is
    equivalent to :code:`x.sqrt()`.

    .. rubric:: Constructors
    ..
       For some reason Sphinx includes __init__ before printing the full table
       of methods.  Not what I want, but I can't figure out how to fix it.  Give
       it its own section so it looks intentional. :)
    """

    def __init__(self, handle, shape, dtype: tl.dtype, space: address_space = None):
        """Not called by user code."""
        # IR handle
        super().__init__(handle)
        self.dtype = dtype.scalar
        self.shape = shape
        self.space = space

    def __str__(self) -> str:
        # ex. "<16x32xfloat32, address_space>"
        res = '<' + 'x'.join(str(s)
                             for s in self.shape) + 'x' + str(self.dtype)
        if self.space:
            res += ', ' + str(self.space)
        return res + '>'
    
    @builtin
    def subview(
        self,
        offsets: List[tl.constexpr],
        sizes: List[tl.constexpr],
        strides: List[tl.constexpr],
        _builder=None
    ) -> 'buffer':
        return semantic.subview(self, offsets, sizes, strides, _builder)

    @builtin
    def to_tensor(self, writable=True, _builder=None):
        """Convert this buffer to a tl.tensor"""
        return to_tensor(self, writable=writable, _builder=_builder)


from . import semantic as semantic # NOQA: E402


@builtin
def allocate_local_buffer(
    type: tl.dtype,
    shape: List[tl.constexpr],
    address_space: address_space = None,
    _builder=None
) -> buffer:
    """
    Allocates a region of local memory with the specified shape and type.

    :param type: the element type of the buffer.
    :type type: tl.dtype
    :param shape: A list of non-negative integers representing the shape of the buffer.
    :type shape: List[tl.constexpr]
    :param address_space: (Optional) backend-specific local memory address space
    :type address_space: bl.address_space
    """
    return semantic.allocate_local_buffer(
        type, shape, address_space, _builder
    )


@builtin
def to_buffer(
    tensor: tl.tensor,
    space: address_space = None,
    _builder=None
) -> buffer:
    """
    Convert a tensor to a buffer.

    :param tensor: the tensor to convert.
    :type tensor: tl.tensor
    :param space: the address space for the buffer (optional).
    :type space: address_space
    """
    return semantic.to_buffer(
        tensor, space, _builder
    )


@builtin
def to_tensor(
    memref: buffer,
    writable: bool = True,
    _builder=None
) -> tl.tensor:
    """
    Create a tl.tensor from a bl.buffer.

    :param memref: the input bl.buffer object.
    :memref type: bl.buffer
    :param writable: If set true, the resultant tensor is considered "writable" during bufferization.
    :type writable: bool
    """
    return semantic.to_tensor(memref, writable, _builder)


@builtin
def subview(
    src: buffer,
    offsets: List[tl.constexpr],
    sizes: List[tl.constexpr],
    strides: List[tl.constexpr],
    _builder=None
) -> buffer:
    '''
    Creates a subview of the source buffer with the specified offsets, sizes, and strides.

    :param src: The source buffer to create a subview from.
    :type src: buffer
    :param offsets: A list of non-negative integers representing the offsets in each dimension.
    :type offsets: List[tl.constexpr]
    :param sizes: A list of non-negative integers representing the sizes in each dimension.
    :type sizes: List[tl.constexpr]
    :param strides: A list of non-negative integers representing the strides in each dimension.
    :type strides: List[tl.constexpr]
    :return: A new buffer representing the subview of the source buffer.
    :rtype: buffer
    '''
    new_offsets = [
        real_semantic.to_tensor(o, _builder) if isinstance(o, tl.constexpr) else o
        for o in offsets
    ]
    return semantic.subview(src, new_offsets, sizes, strides, _builder)
