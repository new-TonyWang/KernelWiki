from triton.language import core, math
from triton.language.standard import max, sum
from triton.runtime.jit import jit

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
    if x.dtype == torch.uint32:
        raise TypeError(f"Unexpected dtype {x.dtype}")
    core.static_assert(_is_power_of_two(x.shape[_get_flip_dim(dim, x.shape)]))
    core.static_assert(_is_power_of_two(x.numel))
    # # reshape the tensor to have all dimensions be 2.
    # # TODO: We shouldn't have to change the dimensions not sorted.
    steps: core.constexpr = _log2(x.numel)
    start: core.constexpr = _log2(x.numel) - _log2(x.shape[_get_flip_dim(dim, x.shape)])
    y = core.reshape(x, [2] * steps)
    y = core.expand_dims(y, start)
    flip = (core.arange(0, 2)[:, None] == 1 - core.arange(0, 2))
    for i in core.static_range(start, steps):
        flip2 = flip
        for j in core.static_range(0, steps + 1):
            if j != i and j != i + 1:
                flip2 = core.expand_dims(flip2, j)
        y = sum(y * flip2, i + 1, keep_dims=True)
    x = core.reshape(y, x.shape)
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
