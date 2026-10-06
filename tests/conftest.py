import devsim_openblas

# DEVSIM reads DEVSIM_MATH_LIBS once, at import, so configure before any test
# module imports it.
devsim_openblas.configure()
