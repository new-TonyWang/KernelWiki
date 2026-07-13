from triton.language import core, math
from triton.language.standard import max, sum
from triton.runtime.jit import jit
from triton.language.standard import (
    _argmax_combine_tie_break_left,
    _argmax_combine_tie_break_fast,
    _elementwise_max,
    _argmin_combine_tie_break_left,
    _argmin_combine_tie_break_fast,
    _elementwise_min,
)

@core._tensor_member_fn
@jit
def flip(x, dim=None):
    """
    Flips a tensor `x` along the dimension `dim`.

    :param x: the first input tensor
    :type x: Block
    :param dim: the dimension to flip along (currently only final dimension supported)
    :type dim: int
    """
    core.static_print("tl.flip is unsupported for now. Use libdevice.flip instead.")
    core.static_assert(False)
    return x

@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("sigmoid")
def sigmoid(x):
    assert core.constexpr(x.dtype.is_floating()), "Unexpected dtype"
    return 1 / (1 + math.exp(-x))

@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding=False):
    assert core.constexpr(x.dtype.is_floating()), "Unexpected dtype"
    z = x - max(x, 0)
    num = math.exp(z)
    den = sum(num, 0)
    return math.fdiv(num, den, ieee_rounding)

@core._tensor_member_fn
@jit
@core._add_reduction_docstr("maximum", return_indices_arg="return_indices",
                            tie_break_arg="return_indices_tie_break_left")
def max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    if core.constexpr(input.dtype.is_bool()) or core.constexpr(input.dtype.is_uint32()):
        raise TypeError(f"Unexpected dtype {input.dtype}")
    input = core._promote_bfloat16_to_float32(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_left, keep_dims=keep_dims)
        else:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_fast, keep_dims=keep_dims)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < core.constexpr(32):
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_int(), "Expecting input to be integer type"
                input = input.to(core.int32)
        return core.reduce(input, axis, _elementwise_max, keep_dims=keep_dims)

@core._tensor_member_fn
@jit
@core._add_reduction_docstr("minimum", return_indices_arg="return_indices",
                            tie_break_arg="return_indices_tie_break_left")
def min(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    if core.constexpr(input.dtype.is_bool()) or core.constexpr(input.dtype.is_uint32()):
        raise TypeError(f"Unexpected dtype {input.dtype}")
    input = core._promote_bfloat16_to_float32(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmin_combine_tie_break_left, keep_dims=keep_dims)
        else:
            return core._reduce_with_indices(input, axis, _argmin_combine_tie_break_fast, keep_dims=keep_dims)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < 32:
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_int(), "Expecting input to be integer type"
                input = input.to(core.int32)
        return core.reduce(input, axis, _elementwise_min, keep_dims=keep_dims)
