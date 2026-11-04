// Compiler hints shared by the tuned kernels.
#pragma once

// Promises the compiler that pointers do not alias, so it can vectorize without runtime checks.
#define SSOLVE_RESTRICT __restrict__

// Keeps a kernel as a separate unit, as with Eigen's solver boundary.
#define SSOLVE_NOINLINE __attribute__((noinline))
