// Method 5: conjugate gradient with Eigen's incomplete Cholesky, its two bugs corrected.
//
// Its own executable, so a VTune run of it sees this method's code only (see method_main.hpp).
#include "method_main.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/cg.hpp"
#include "ssolve/preconditioner.hpp"

using namespace ssolve;

int main(int argc, char** argv) {
  return ssolve_methods::method_main(
      argc, argv, "cg_ic_fixed", "CG + incomplete Cholesky (fixed), CSR", [](const CsrMatrix& a, const double* b, double* x, const SolveOptions& options) {
        const double setup_start = wall_seconds();
        const auto backend = make_seq_backend(clone(a));
        const auto precond = make_preconditioner(Precond::IncompleteCholeskyFixed, a);
        const double setup_seconds = wall_seconds() - setup_start;

        SolveResult result;
        {
          const ssolve_methods::ProfiledRegion region;
          result = cg(*backend, *precond, b, x, options);
        }
        result.setup_seconds = setup_seconds;
        verify_result(a, b, x, options, result);
        return result;
      });
}
