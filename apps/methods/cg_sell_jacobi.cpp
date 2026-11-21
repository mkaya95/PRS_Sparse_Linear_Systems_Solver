// Method 2: conjugate gradient with the Jacobi preconditioner, matrix in SELL-C-sigma (C = 8, sigma = 256).
//
// Run as CG on the scaled system S A S (S = D^(-1/2)), whose unit diagonal is not stored: the same
// iterates as Jacobi-preconditioned CG, with fewer bytes per iteration (see ssolve/jacobi_scaled.hpp).
// Its own executable, so a VTune run of it sees this method's code only (see method_main.hpp).
#include "method_main.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/jacobi_scaled.hpp"

using namespace ssolve;

int main(int argc, char** argv) {
  return ssolve_methods::method_main(
      argc, argv, "cg_sell_jacobi", "CG + Jacobi, SELL-8-256", [](const CsrMatrix& a, const double* b, double* x, const SolveOptions& options) {
        const double setup_start = wall_seconds();
        const JacobiScaling scaling = jacobi_scaling(a);
        const auto backend = make_sell_backend_unit_diagonal(scaling.off_diagonal);
        const double setup_seconds = wall_seconds() - setup_start;

        SolveResult result;
        {
          const ssolve_methods::ProfiledRegion region;
          result = cg_scaled_jacobi(*backend, scaling, b, x, options);
        }
        result.setup_seconds = setup_seconds;
        verify_result(a, b, x, options, result);
        return result;
      });
}
