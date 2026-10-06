# devsim-openblas

[DEVSIM](https://devsim.org) loads BLAS/LAPACK at import time from whatever
`DEVSIM_MATH_LIBS` names, falling back to Intel MKL or a system
`libopenblas`. Pip alone cannot provide either on every platform: MKL wheels
cover Linux x86_64 and Windows x64 only, and the portable OpenBLAS wheel,
`scipy-openblas32`, prefixes its symbols (`scipy_dgetrf_`), so DEVSIM cannot
find them.

This package ships a tiny shim library that exports the six routines DEVSIM
looks up (`dgetrf_`, `dgetrs_`, `zgetrf_`, `zgetrs_`, `drotg_`, `zrotg_`) and
forwards them to `scipy-openblas32`.

```python
import devsim_openblas

devsim_openblas.configure()  # before the first `import devsim`
import devsim
```

`configure()` overrides any `DEVSIM_MATH_LIBS` already set. MKL PARDISO is
not available through the shim; DEVSIM's bundled SuperLU/UMFPACK direct
solvers are.
