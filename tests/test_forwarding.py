"""Each forwarded routine gives OpenBLAS's answer when called through the shim."""

import ctypes

import numpy as np
import pytest

import devsim_openblas

shim = ctypes.CDLL(str(devsim_openblas.shim_path()))
c_int_p = ctypes.POINTER(ctypes.c_int)


def _int(value):
    return ctypes.byref(ctypes.c_int(value))


@pytest.mark.parametrize(("dtype", "prefix"), [(np.float64, "d"), (np.complex128, "z")])
def test_getrf_getrs_solve(dtype, prefix):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 5)).astype(dtype)
    b = rng.standard_normal(5).astype(dtype)
    if dtype is np.complex128:
        a = a + 1j * rng.standard_normal((5, 5))
        b = b + 1j * rng.standard_normal(5)
    lu = np.asfortranarray(a.copy())
    x = b.copy()
    ipiv = np.zeros(5, dtype=np.int32)
    info = ctypes.c_int(-99)

    getattr(shim, f"{prefix}getrf_")(
        _int(5), _int(5), lu.ctypes.data_as(ctypes.c_void_p), _int(5),
        ipiv.ctypes.data_as(c_int_p), ctypes.byref(info),
    )
    assert info.value == 0
    getattr(shim, f"{prefix}getrs_")(
        ctypes.c_char_p(b"N"), _int(5), _int(1), lu.ctypes.data_as(ctypes.c_void_p),
        _int(5), ipiv.ctypes.data_as(c_int_p), x.ctypes.data_as(ctypes.c_void_p),
        _int(5), ctypes.byref(info),
    )
    assert info.value == 0
    np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-12)


def test_drotg():
    a, b, c, s = (ctypes.c_double(v) for v in (3.0, 4.0, 0.0, 0.0))
    shim.drotg_(*(ctypes.byref(v) for v in (a, b, c, s)))
    assert (a.value, c.value, s.value) == pytest.approx((5.0, 0.6, 0.8))


def test_zrotg():
    a = np.array([3.0 + 0j])
    b = np.array([4.0 + 0j])
    c = np.zeros(1)
    s = np.zeros(1, dtype=np.complex128)
    shim.zrotg_(*(v.ctypes.data_as(ctypes.c_void_p) for v in (a, b, c, s)))
    assert a[0] == pytest.approx(5.0)
    assert c[0] == pytest.approx(0.6)
    assert s[0] == pytest.approx(0.8)


def test_dgemm():
    rng = np.random.default_rng(1)
    a = np.asfortranarray(rng.standard_normal((3, 4)))
    b = np.asfortranarray(rng.standard_normal((4, 2)))
    c = np.asfortranarray(np.zeros((3, 2)))
    one, zero = ctypes.c_double(1.0), ctypes.c_double(0.0)
    ptr = lambda arr: arr.ctypes.data_as(ctypes.c_void_p)
    shim.dgemm_(
        ctypes.c_char_p(b"N"), ctypes.c_char_p(b"N"), _int(3), _int(2), _int(4),
        ctypes.byref(one), ptr(a), _int(3), ptr(b), _int(4),
        ctypes.byref(zero), ptr(c), _int(3),
    )
    np.testing.assert_allclose(c, a @ b, rtol=1e-12)
