"""Repair NVRTC include lookup when the Python environment path is non-ASCII."""
from __future__ import annotations

from pathlib import Path
import os
import tempfile


def install_ascii_include_alias(cupy_module, alias: Path = Path(os.environ.get("NEB_CUPY_ASCII_HEADERS", str(Path(tempfile.gettempdir()) / "neb_cupy_headers")))) -> None:
    import cupy.cuda.compiler as compiler

    if getattr(compiler._NVRTCProgram.compile, "_eigenfluid_ascii_include", False):
        return
    header_root = Path(cupy_module.__file__).resolve().parent / "_core" / "include"
    if str(header_root).isascii():
        return
    if not alias.exists() or alias.resolve() != header_root.resolve():
        raise RuntimeError(
            f"NVRTC ASCII include junction {alias} must target {header_root}"
        )
    original = compiler._NVRTCProgram.compile

    def compile_with_ascii_headers(self, options=(), *args, **kwargs):
        rewritten = []
        marker = "cupy\\_core\\include"
        for option in options:
            if marker in option:
                suffix = option.split(marker, 1)[1].replace("/", "\\")
                option = f"-I{alias}{suffix}"
            rewritten.append(option)
        return original(self, tuple(rewritten), *args, **kwargs)

    compile_with_ascii_headers._eigenfluid_ascii_include = True
    compiler._NVRTCProgram.compile = compile_with_ascii_headers
