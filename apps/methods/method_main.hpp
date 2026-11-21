// What every one-method executable shares: reading the matrix, building b, timing and printing.
//
// Each of the nine methods has its own executable (apps/methods/*.cpp) with its own main, so VTune profiles
// one method's code at a time and nothing of the others. The method file builds its storage and its
// preconditioner, then runs the iterations inside a ProfiledRegion.
//
// Usage: <method> --matrix SPEC [--tol 1e-8] [--maxit N] [--problem random|a1|unit] [--repeat 1]
//
// With VTune started paused (vtune -start-paused ...), only the code inside ProfiledRegion is collected:
// the iterations, not the reading of the file or the setup. Without VTune the region costs nothing.
#pragma once

#include <cstdio>
#include <exception>
#include <functional>
#include <string>

#include "../cli.hpp"
#include "../problem.hpp"
#include "ssolve/aligned_array.hpp"
#include "ssolve/matrix.hpp"
#include "ssolve/matrix_spec.hpp"
#include "ssolve/solver.hpp"
#include "ssolve/timer.hpp"
#include "ssolve/verify.hpp"

#ifdef SSOLVE_ITT
#include <ittnotify.h>
#endif

namespace ssolve_methods {

// Tells VTune to collect from construction to destruction. VTune must have been started with -start-paused.
class ProfiledRegion {
 public:
  ProfiledRegion() {
#ifdef SSOLVE_ITT
    __itt_resume();
#endif
  }
  ~ProfiledRegion() {
#ifdef SSOLVE_ITT
    __itt_pause();
#endif
  }
  ProfiledRegion(const ProfiledRegion&) = delete;
  ProfiledRegion& operator=(const ProfiledRegion&) = delete;
};

// One method: builds what it needs from a (setup, timed into setup_seconds), then solves inside a
// ProfiledRegion. The returned result's solve_seconds covers only the iterations.
using Method = std::function<ssolve::SolveResult(const ssolve::CsrMatrix& a, const double* b, double* x,
                                                 const ssolve::SolveOptions& options)>;

inline int method_main(int argc, char** argv, const char* name, const char* description, const Method& method) {
  using namespace ssolve;
  try {
    const ssolve_cli::Args args(argc, argv, {"matrix", "tol", "maxit", "problem", "repeat"});
    SolveOptions options;
    options.tol = args.get_double("tol", 1e-8);
    options.max_iterations = args.get_int("maxit", 0);
    validate_options(options);
    const int repeats = args.get_count("repeat", 1, 1);
    const std::string spec = args.required("matrix");

    const CsrMatrix a = load_matrix_spec(spec);
    validate_cg_matrix(a);
    const AlignedArray<double> b = ssolve_cli::rhs_for_problem(args.get("problem", "random"), a);
    AlignedArray<double> x(static_cast<std::size_t>(a.n_rows));

    std::printf("method         %s (%s)\n", name, description);
    std::printf("matrix         %s, n = %d, nnz = %lld\n", matrix_display_name(spec).c_str(), a.n_rows,
                static_cast<long long>(a.nnz()));
    int exit_status = 0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      x.fill(0.0);
      const SolveResult result = method(a, b.data(), x.data(), options);
      const double per_iteration =
          result.spmv_count > 0 ? result.solve_seconds / static_cast<double>(result.spmv_count) : 0.0;
      std::printf("run %d          %s, %lld iterations, setup %.3f s, solve %.3f s, %.3f ms per iteration, "
                  "residual %.2e (true %.2e)\n",
                  repeat + 1, to_string(result.status), static_cast<long long>(result.spmv_count),
                  result.setup_seconds, result.solve_seconds, per_iteration * 1e3, result.recursive_residual,
                  result.true_residual);
      if (result.detail[0] != '\0') std::printf("reason         %s\n", result.detail);
      if (result.status != SolveStatus::Converged) exit_status = 2;
    }
    return exit_status;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: %s\n", name, e.what());
    return 1;
  }
}

}  // namespace ssolve_methods
