/*
 * Forward the BLAS/LAPACK symbols DEVSIM resolves at runtime to the prefixed
 * copies exported by scipy-openblas32 (scipy_dgetrf_, ...).
 *
 * DEVSIM dlopens every library listed in DEVSIM_MATH_LIBS and looks up
 * dgetrf_, dgetrs_, zgetrf_, zgetrs_, drotg_ and zrotg_ by name (upstream
 * src/math/BlasHeaders.cc). Its UMFPACK direct solver then opens the same
 * libraries and looks up the level-2/3 BLAS routines below. scipy-openblas32
 * prefixes all of its symbols with scipy_, so DEVSIM cannot use it directly.
 * This library exports the unprefixed names and calls through to the prefixed
 * ones.
 *
 * Every argument of a Fortran BLAS/LAPACK routine is passed by reference, so
 * each forwarder takes and passes plain pointers. Trailing hidden string
 * lengths some callers add are ignored by OpenBLAS's C implementations, and
 * extra trailing arguments are harmless under the cdecl and x64 conventions.
 *
 * OpenBLAS is resolved at runtime by devsim_openblas_shim_init, so the shim
 * has no link-time dependency on scipy-openblas32.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define SHIM_EXPORT __declspec(dllexport)
#else
#include <dlfcn.h>
#define SHIM_EXPORT __attribute__((visibility("default")))
#endif

#define SYMBOLS(X)                                                             \
  /* LAPACK and BLAS-1, loaded by DEVSIM itself */                             \
  X(dgetrf, 6) X(dgetrs, 9) X(zgetrf, 6) X(zgetrs, 9) X(drotg, 4) X(zrotg, 4)  \
  /* BLAS-2/3, loaded by DEVSIM's UMFPACK plugin */                            \
  X(dgemm, 13) X(dgemv, 11) X(dger, 9) X(dtrsm, 11) X(dtrsv, 8)               \
  X(zgemm, 13) X(zgemv, 11) X(zgeru, 9) X(ztrsm, 11) X(ztrsv, 8)

#define AS_ENUM(name, nargs) SYM_##name,
enum { SYMBOLS(AS_ENUM) NSYMBOLS };

#define AS_NAME(name, nargs) #name,
static const char *const names[NSYMBOLS] = {SYMBOLS(AS_NAME)};

static void *targets[NSYMBOLS];
static unsigned long calls[NSYMBOLS];
static char last_error[512];

static void *resolve(int which) {
  void *f = targets[which];
  if (!f) {
    fprintf(stderr,
            "devsim-openblas: %s_ called before devsim_openblas.configure() "
            "loaded OpenBLAS\n",
            names[which]);
    abort();
  }
  calls[which]++;
  return f;
}

typedef void *P;
#define PARAMS_4 P a1, P a2, P a3, P a4
#define PARAMS_6 PARAMS_4, P a5, P a6
#define PARAMS_8 PARAMS_6, P a7, P a8
#define PARAMS_9 PARAMS_8, P a9
#define PARAMS_11 PARAMS_9, P a10, P a11
#define PARAMS_13 PARAMS_11, P a12, P a13
#define ARGS_4 a1, a2, a3, a4
#define ARGS_6 ARGS_4, a5, a6
#define ARGS_8 ARGS_6, a7, a8
#define ARGS_9 ARGS_8, a9
#define ARGS_11 ARGS_9, a10, a11
#define ARGS_13 ARGS_11, a12, a13

#define AS_FORWARDER(name, nargs)                                              \
  SHIM_EXPORT void name##_(PARAMS_##nargs) {                                   \
    ((void (*)(PARAMS_##nargs))resolve(SYM_##name))(ARGS_##nargs);             \
  }
SYMBOLS(AS_FORWARDER)

SHIM_EXPORT int devsim_openblas_shim_init(const char *openblas_path) {
  void *handle;
  char symbol[32];
  int missing = 0;
  int i;

  last_error[0] = '\0';
#ifdef _WIN32
  handle = (void *)LoadLibraryExA(openblas_path, NULL,
                                  LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!handle) {
    snprintf(last_error, sizeof last_error,
             "LoadLibrary(\"%s\") failed with error %lu", openblas_path,
             (unsigned long)GetLastError());
    return -1;
  }
#else
  handle = dlopen(openblas_path, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    snprintf(last_error, sizeof last_error, "%s", dlerror());
    return -1;
  }
#endif

  for (i = 0; i < NSYMBOLS; ++i) {
    snprintf(symbol, sizeof symbol, "scipy_%s_", names[i]);
#ifdef _WIN32
    targets[i] = (void *)GetProcAddress((HMODULE)handle, symbol);
#else
    targets[i] = dlsym(handle, symbol);
#endif
    if (!targets[i]) {
      size_t used = strlen(last_error);
      snprintf(last_error + used, sizeof last_error - used, "%s%s",
               used ? ", " : "missing symbols: ", symbol);
      ++missing;
    }
  }
  return missing;
}

SHIM_EXPORT const char *devsim_openblas_shim_error(void) { return last_error; }

/* Name of forwarded routine i without the trailing underscore, NULL past the end. */
SHIM_EXPORT const char *devsim_openblas_shim_name(int i) {
  return (i >= 0 && i < NSYMBOLS) ? names[i] : NULL;
}

SHIM_EXPORT unsigned long devsim_openblas_shim_calls(int i) {
  return (i >= 0 && i < NSYMBOLS) ? calls[i] : 0;
}
