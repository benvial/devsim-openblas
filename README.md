# devsim-openblas

Run [DEVSIM](https://devsim.org) on the OpenBLAS that NumPy and SciPy ship, so
`pip install devsim devsim-openblas` gives a working DEVSIM on Linux, macOS
and Windows with no system BLAS/LAPACK and no Intel MKL.

```python
import devsim_openblas

devsim_openblas.configure()  # before the first `import devsim`
import devsim
```

## Why

DEVSIM loads BLAS/LAPACK at import time from whatever `DEVSIM_MATH_LIBS`
names, falling back to Intel MKL or a system `libopenblas`. Pip alone cannot
provide either on every platform: MKL wheels cover Linux x86_64 and Windows
x64 only (and DEVSIM does not load MKL on macOS), while the portable OpenBLAS
wheel, [`scipy-openblas32`](https://pypi.org/project/scipy-openblas32/),
prefixes its symbols (`scipy_dgetrf_`), so DEVSIM cannot find them.

This package ships a small shim library that exports, under their plain
names, the 16 routines DEVSIM and its UMFPACK solver look up, and forwards
each to its `scipy_`-prefixed counterpart:

- LAPACK and BLAS-1, loaded by DEVSIM: `dgetrf_ dgetrs_ zgetrf_ zgetrs_ drotg_ zrotg_`
- BLAS-2/3, loaded by the UMFPACK direct solver: `dgemm_ dgemv_ dger_ dtrsm_ dtrsv_`
  and `zgemm_ zgemv_ zgeru_ ztrsm_ ztrsv_`

`configure()` loads OpenBLAS behind the shim and sets `DEVSIM_MATH_LIBS` to
the shim, overriding any value already set. It must run before DEVSIM is
imported, because DEVSIM reads the variable once. MKL PARDISO is not
available through the shim; DEVSIM's UMFPACK and SuperLU solvers are.

`devsim_openblas.calls()` reports how many times DEVSIM has called each
forwarded routine, which is handy to check the shim is in use.

## Platforms

Wheels are built and tested in CI on every platform DEVSIM publishes wheels
for: Linux x86_64 and aarch64, macOS arm64 and Windows x64. One wheel per
platform serves every Python 3 version.

## Status

This is a stopgap. A fallback that lets DEVSIM load `scipy-openblas32`
directly has been proposed upstream in
[devsim/devsim#167](https://github.com/devsim/devsim/issues/167); once a DEVSIM
release includes it, this package is no longer needed.
