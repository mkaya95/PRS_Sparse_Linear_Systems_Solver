// The optimization ladder: each rung is CG with one more change than the one before.
//
// Old rungs stay here on purpose. They are measurement fixtures, so the results table
// can be regenerated at any time instead of being reconstructed from git history. The
// shipping solver lives in src/ and holds only what the measurements justified.
//
// Kernels are noinline so that each stays a separate unit, as with Eigen's solver boundary.
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "kernels.hpp"
#include "registry.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/cg.hpp"
#include "ssolve/cg_common.hpp"
#include "ssolve/kernels_common.hpp"
#include "ssolve/timer.hpp"

using namespace ssolve;

namespace ssolve_bench {

namespace {

// L0: what anyone writes first. No restrict, no pragmas, one accumulator.
struct KNaive {
  SSOLVE_NOINLINE static void spmv(const CsrMatrix& a, const double* x, double* y) {
    for (Index i = 0; i < a.n_rows; ++i) {
      double sum = 0.0;
      for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) sum += a.val[k] * x[a.col[k]];
      y[i] = sum;
    }
  }
  SSOLVE_NOINLINE static double dot(const double* x, const double* y, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += x[i] * y[i];
    return sum;
  }
  SSOLVE_NOINLINE static void axpy(double alpha, const double* x, double* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
  }
  SSOLVE_NOINLINE static void xpby(const double* x, double beta, double* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i] + beta * y[i];
  }
};

// L1: the same loops, with non-aliasing pointers and local copies of the matrix arrays.
struct KRestrict {
  SSOLVE_NOINLINE static void spmv(const CsrMatrix& a, const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y) {
    const std::int64_t* SSOLVE_RESTRICT const row_ptr = a.row_ptr.data();
    const Index* SSOLVE_RESTRICT const col = a.col.data();
    const double* SSOLVE_RESTRICT const val = a.val.data();
    const Index rows = a.n_rows;
    for (Index i = 0; i < rows; ++i) {
      double sum = 0.0;
      for (std::int64_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) sum += val[k] * x[col[k]];
      y[i] = sum;
    }
  }
  SSOLVE_NOINLINE static double dot(const double* SSOLVE_RESTRICT x, const double* SSOLVE_RESTRICT y, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += x[i] * y[i];
    return sum;
  }
  SSOLVE_NOINLINE static void axpy(double alpha, const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y,
                                   std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
  }
  SSOLVE_NOINLINE static void xpby(const double* SSOLVE_RESTRICT x, double beta, double* SSOLVE_RESTRICT y,
                                   std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i] + beta * y[i];
  }
};

// L2: SIMD on the vector kernels. GCC vectorizes floating-point sums only with this pragma or
// -ffast-math, because adding in a different order can change the rounded result.
struct KSimd : KRestrict {
  SSOLVE_NOINLINE static double dot(const double* SSOLVE_RESTRICT x, const double* SSOLVE_RESTRICT y, std::size_t n) {
    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
    for (std::size_t i = 0; i < n; ++i) sum += x[i] * y[i];
    return sum;
  }
  SSOLVE_NOINLINE static void axpy(double alpha, const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y,
                                   std::size_t n) {
#pragma omp simd
    for (std::size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
  }
  SSOLVE_NOINLINE static void xpby(const double* SSOLVE_RESTRICT x, double beta, double* SSOLVE_RESTRICT y,
                                   std::size_t n) {
#pragma omp simd
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i] + beta * y[i];
  }
};

// L3: three candidates for the CSR row loop. Only spmv changes.
//
// Two accumulators, as in Eigen. Breaking the dependency between consecutive additions
// lets the CPU run two of them at once.
struct KAcc2 : KSimd {
  SSOLVE_NOINLINE static void spmv(const CsrMatrix& a, const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y) {
    const std::int64_t* SSOLVE_RESTRICT const row_ptr = a.row_ptr.data();
    const Index* SSOLVE_RESTRICT const col = a.col.data();
    const double* SSOLVE_RESTRICT const val = a.val.data();
    const Index rows = a.n_rows;
    for (Index i = 0; i < rows; ++i) {
      double sum0 = 0.0, sum1 = 0.0;
      std::int64_t k = row_ptr[i];
      const std::int64_t end = row_ptr[i + 1];
      for (; k + 1 < end; k += 2) {
        sum0 += val[k] * x[col[k]];
        sum1 += val[k + 1] * x[col[k + 1]];
      }
      if (k < end) sum0 += val[k] * x[col[k]];
      y[i] = sum0 + sum1;
    }
  }
};

// Four accumulators with a remainder loop.
struct KUnroll4 : KSimd {
  SSOLVE_NOINLINE static void spmv(const CsrMatrix& a, const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y) {
    const std::int64_t* SSOLVE_RESTRICT const row_ptr = a.row_ptr.data();
    const Index* SSOLVE_RESTRICT const col = a.col.data();
    const double* SSOLVE_RESTRICT const val = a.val.data();
    const Index rows = a.n_rows;
    for (Index i = 0; i < rows; ++i) {
      double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
      std::int64_t k = row_ptr[i];
      const std::int64_t end = row_ptr[i + 1];
      for (; k + 3 < end; k += 4) {
        sum0 += val[k] * x[col[k]];
        sum1 += val[k + 1] * x[col[k + 1]];
        sum2 += val[k + 2] * x[col[k + 2]];
        sum3 += val[k + 3] * x[col[k + 3]];
      }
      for (; k < end; ++k) sum0 += val[k] * x[col[k]];
      y[i] = (sum0 + sum1) + (sum2 + sum3);
    }
  }
};

// SIMD over each row, using gather instructions for x.
struct KSimdRow : KSimd {
  SSOLVE_NOINLINE static void spmv(const CsrMatrix& a, const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y) {
    const std::int64_t* SSOLVE_RESTRICT const row_ptr = a.row_ptr.data();
    const Index* SSOLVE_RESTRICT const col = a.col.data();
    const double* SSOLVE_RESTRICT const val = a.val.data();
    const Index rows = a.n_rows;
    for (Index i = 0; i < rows; ++i) {
      double sum = 0.0;
      const std::int64_t begin = row_ptr[i];
      const std::int64_t end = row_ptr[i + 1];
#pragma omp simd reduction(+ : sum)
      for (std::int64_t k = begin; k < end; ++k) sum += val[k] * x[col[k]];
      y[i] = sum;
    }
  }
};

// Textbook CG: one separate pass per operation, as in Eigen's loop.
template <class K>
SolveResult cg_passes(const CsrMatrix& a, const double* b, double* x, const SolveOptions& options) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("CG needs a square matrix");
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const double start = wall_seconds();

  SolveResult result = with_rhs_scaling(n, b, x, options, [&](const double* rhs, double* solution) {
    const std::int64_t limit =
        options.max_iterations > 0 ? options.max_iterations : 2 * static_cast<std::int64_t>(n);
    AlignedArray<double> r_array(n), p_array(n), q_array(n);
    double* const r = r_array.data();
    double* const p = p_array.data();
    double* const q = q_array.data();

    if (!options.use_initial_guess) {
      for (std::size_t i = 0; i < n; ++i) solution[i] = 0.0;
    }
    K::spmv(a, solution, q);
    for (std::size_t i = 0; i < n; ++i) r[i] = rhs[i] - q[i];
    for (std::size_t i = 0; i < n; ++i) p[i] = r[i];

    const double rhs_squared = K::dot(rhs, rhs, n);
    const double threshold =
        std::max(options.tol * options.tol * rhs_squared, std::numeric_limits<double>::min());
    double rr = K::dot(r, r, n);

    SolveResult out;
    const auto finish = [&](SolveStatus status) {
      out.status = status;
      out.recursive_residual = std::sqrt(rr / rhs_squared);
      return out;
    };
    if (rr < threshold) return finish(SolveStatus::Converged);

    while (out.spmv_count < limit) {
      K::spmv(a, p, q);
      ++out.spmv_count;
      const double pq = K::dot(p, q, n);
      if (!(pq > 0.0)) return finish(SolveStatus::Breakdown);  // also catches NaN
      const double alpha = rr / pq;
      K::axpy(alpha, p, solution, n);
      K::axpy(-alpha, q, r, n);
      const double rr_new = K::dot(r, r, n);
      if (rr_new < threshold) {
        rr = rr_new;
        return finish(SolveStatus::Converged);
      }
      K::xpby(r, rr_new / rr, p, n);
      rr = rr_new;
    }
    return finish(SolveStatus::MaxIterations);
  });
  result.solve_seconds = wall_seconds() - start;
  return result;
}

// The shipping solver, as a ladder rung. The copy of the matrix happens before any timer,
// and building the backend counts as setup, like Eigen's compute().
template <bool AllowNarrowRowPtr, bool InitialSpmv, Precond P>
SolveResult run_backend(const CsrMatrix& a, const double* b, double* x, const SolveOptions& options) {
  CsrMatrix copy = clone(a);
  const double setup_start = wall_seconds();
  const std::unique_ptr<Backend> backend = make_seq_backend(std::move(copy), AllowNarrowRowPtr);
  AlignedArray<double> inv_diag;
  if (P == Precond::Jacobi) inv_diag = jacobi_inverse_diagonal(a);
  const double setup_seconds = wall_seconds() - setup_start;

  SolveOptions used = options;
  if (InitialSpmv && !options.use_initial_guess) {
    // A zero guess makes cg() compute r = b - A x with an SpMV, as the earlier rungs do.
    backend->set_zero(x);
    used.use_initial_guess = true;
  }
  SolveResult result = cg(*backend, P == Precond::Jacobi ? inv_diag.data() : nullptr, b, x, used);
  result.setup_seconds = setup_seconds;
  return result;
}

template <class K>
KernelSet make_kernels(const CsrMatrix& a) {
  const CsrMatrix* const matrix = &a;
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  KernelSet set;
  set.spmv = [matrix](const double* x, double* y) { K::spmv(*matrix, x, y); };
  set.dot = [n](const double* x, const double* y) { return K::dot(x, y, n); };
  set.axpy = [n](double alpha, const double* x, double* y) { K::axpy(alpha, x, y, n); };
  set.norm2 = [n](const double* x) { return K::dot(x, x, n); };
  set.xpby = [n](const double* x, double beta, double* y) { K::xpby(x, beta, y, n); };
  return set;
}

}  // namespace

std::vector<std::string> ladder_kernel_names() {
  return {"L0", "L1", "L2", "L3-acc2", "L3-unroll4", "L3-simdrow"};
}

KernelSet ladder_kernels(const std::string& name, const CsrMatrix& a) {
  if (name == "L0") return make_kernels<KNaive>(a);
  if (name == "L1") return make_kernels<KRestrict>(a);
  if (name == "L2") return make_kernels<KSimd>(a);
  if (name == "L3-acc2") return make_kernels<KAcc2>(a);
  if (name == "L3-unroll4") return make_kernels<KUnroll4>(a);
  if (name == "L3-simdrow") return make_kernels<KSimdRow>(a);
  std::string names = "eigen";
  for (const std::string& valid : ladder_kernel_names()) names += ", " + valid;
  throw std::runtime_error("unknown implementation '" + name + "'. Valid names: " + names);
}

std::vector<NamedSolver> ladder_solvers() {
  return {
      {"L0-naive", Precond::None, cg_passes<KNaive>},
      {"L1-restrict", Precond::None, cg_passes<KRestrict>},
      {"L2-simd", Precond::None, cg_passes<KSimd>},
      {"L3-acc2", Precond::None, cg_passes<KAcc2>},
      {"L3-unroll4", Precond::None, cg_passes<KUnroll4>},
      {"L3-simdrow", Precond::None, cg_passes<KSimdRow>},
      {"L4-fused", Precond::None, run_backend<false, true, Precond::None>},
  };
}

}  // namespace ssolve_bench
