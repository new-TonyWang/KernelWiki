from triton.language.core import _constexpr_to_value
from triton._C.libtriton import ir

class Scope:
    def __init__(self, core_mode: str, builder=None):
        self.core_mode = constexpr_to_value(core_mode) if _builder is None else core_mode
        self._builder = builder
        if(self.core_mode not in ['cube', 'vector']):
            raise ValueError("Not supported core_mode")

    def __enter__(self):
        self._builder.core_mode_stack.append(self.core_mode)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._builder.core_mode_stack.pop()
