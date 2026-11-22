// Method 6: conjugate gradient with IC(0), incomplete Cholesky keeping exactly the pattern of A.
//
// Its own executable, so a VTune run of it sees this method's code only (see method_main.hpp).
#include "method_main.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/cg.hpp"
#include "ssolve/preconditioner.hpp"

using namespace ssolve;

int main(int argc, char** argv) {
  return ssolve_methods::method_main(
      argc, argv, "cg_ic0", "CG + IC(0), CSR", [](const CsrMatrix& a, const double* b, double* x, const SolveOptions& options) {
        const double setup_start = wall_seconds();
        const auto backend = make_seq_backend(clone(a));
        const auto precond = make_preconditioner(Precond::IncompleteCholeskyZero, a);
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
