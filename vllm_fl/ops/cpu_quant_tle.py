"""Inductor source integration for ARM quantized TLE builtins."""

from threading import Lock


_LOCK = Lock()
_BUILTIN_IMPORTS: dict[str, str] = {}
_INDUCTOR_PATCHED = False


def register_inductor_builtin_import(trigger: str, import_line: str) -> None:
    """Register one custom builtin import with a single Inductor source patch."""
    import torch._inductor.async_compile as async_compile

    global _INDUCTOR_PATCHED
    with _LOCK:
        previous = _BUILTIN_IMPORTS.get(trigger)
        if previous is not None and previous != import_line:
            raise RuntimeError(f"conflicting Inductor import for {trigger}")
        _BUILTIN_IMPORTS[trigger] = import_line
        if _INDUCTOR_PATCHED:
            return

        original_triton = async_compile.AsyncCompile.triton

        def compile_triton(self, kernel_name, source_code, device_str="cpu"):
            with _LOCK:
                registered = tuple(_BUILTIN_IMPORTS.items())
            imports = [
                line
                for marker, line in registered
                if marker in source_code and line.strip() not in source_code
            ]
            if imports:
                source_code = "".join(imports) + source_code
            return original_triton(self, kernel_name, source_code, device_str)

        async_compile.AsyncCompile.triton = compile_triton
        _INDUCTOR_PATCHED = True
