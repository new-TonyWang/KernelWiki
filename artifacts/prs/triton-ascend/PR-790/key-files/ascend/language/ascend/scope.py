# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

from triton._C.libtriton import ir

# Handle constexpr values
def _constexpr_to_value(val):
    """Convert constexpr to value if needed"""
    if hasattr(val, 'value'):
        return val.value
    return val


class Scope:
    """
    Context manager to set the core mode (cube or vector) for operations within the scope.
    
    This allows you to explicitly specify which core type should be used for operations
    within a code block, helping the compiler generate appropriate code for cube or vector cores.
    
    Example:
        import triton.language.ascend as al
        
        @triton.jit
        def kernel(x_ptr, y_ptr, N):
            with al.Scope(core_mode="cube"):
                # Operations here will be marked for cube core
                result = tl.dot(a, b)
            
            with al.Scope(core_mode="vector"):
                # Operations here will be marked for vector core
                result = a + b
    """
    def __init__(self, core_mode: str, _builder=None, _semantic=None, **kwargs):
        """
        :param core_mode: Either "cube" or "vector" to specify the core type
        :param _builder: Internal builder object (set by code_generator)
        :param _semantic: Internal semantic object (set by code_generator)
        :param kwargs: Additional internal parameters
        """
        self.core_mode = _constexpr_to_value(core_mode) if _builder is None else core_mode
        self._builder = _builder
        self._semantic = _semantic
        if self.core_mode not in ("cube", "vector"):
            raise ValueError(f'core_mode must be "cube" or "vector", got {self.core_mode}')
    
    def __enter__(self):
        if self._builder is None:
            raise RuntimeError("Scope can only be used inside a Triton kernel")
        # Store the current core_mode in the builder
        if not hasattr(self._builder, 'core_mode_stack'):
            self._builder.core_mode_stack = []
        self._builder.core_mode_stack.append(self.core_mode)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._builder, 'core_mode_stack') and self._builder.core_mode_stack:
            self._builder.core_mode_stack.pop()
        return False

