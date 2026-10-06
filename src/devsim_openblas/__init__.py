"""Point DEVSIM at the OpenBLAS shipped by scipy-openblas32.

Call :func:`configure` before the first ``import devsim``::

    import devsim_openblas

    devsim_openblas.configure()
    import devsim
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import scipy_openblas32

__all__ = ["calls", "configure", "openblas_path", "shim_path"]

_SUFFIX = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
_shim: ctypes.CDLL | None = None


def shim_path() -> Path:
    """Path of the shim library DEVSIM_MATH_LIBS should name."""
    (path,) = Path(__file__).parent.glob(f"*devsim_openblas_shim{_SUFFIX}")
    return path


def openblas_path() -> Path:
    """Path of the scipy-openblas32 shared library."""
    lib_dir = Path(scipy_openblas32.get_lib_dir())
    (path,) = lib_dir.glob(f"libscipy_openblas*{_SUFFIX}")
    return path


def configure() -> Path:
    """Load OpenBLAS behind the shim and set DEVSIM_MATH_LIBS to the shim.

    Must run before DEVSIM is imported: DEVSIM reads DEVSIM_MATH_LIBS once, at
    import. Overrides any DEVSIM_MATH_LIBS already set. Returns the shim path.
    """
    global _shim
    if "devsim" in sys.modules:
        raise RuntimeError(
            "devsim_openblas.configure() must run before DEVSIM is imported"
        )
    path = shim_path()
    if _shim is None:
        # DEVSIM later opens the same path and gets this already-initialised
        # copy of the library back from the loader.
        shim = ctypes.CDLL(str(path))
        shim.devsim_openblas_shim_init.argtypes = [ctypes.c_char_p]
        shim.devsim_openblas_shim_init.restype = ctypes.c_int
        shim.devsim_openblas_shim_error.restype = ctypes.c_char_p
        if shim.devsim_openblas_shim_init(os.fsencode(openblas_path())) != 0:
            error = shim.devsim_openblas_shim_error().decode()
            raise ImportError(f"devsim-openblas could not load OpenBLAS: {error}")
        _shim = shim
    os.environ["DEVSIM_MATH_LIBS"] = str(path)
    return path


def calls() -> dict[str, int]:
    """How many times DEVSIM has called each forwarded routine (diagnostics)."""
    if _shim is None:
        return {}
    _shim.devsim_openblas_shim_name.restype = ctypes.c_char_p
    _shim.devsim_openblas_shim_calls.restype = ctypes.c_ulong
    counts = {}
    i = 0
    while (name := _shim.devsim_openblas_shim_name(i)) is not None:
        counts[name.decode()] = _shim.devsim_openblas_shim_calls(i)
        i += 1
    return counts
