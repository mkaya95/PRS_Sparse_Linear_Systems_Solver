// Method 7: conjugate gradient with IC(0) on the matrix reordered by reverse Cuthill-McKee.
//
// The solve runs on P A Pᵀ (P x) = P b. Reordering is setup; moving b in and x out is inside the solve.
// Its own executable, so a VTune run of it sees this method's code only (see method_main.hpp).
#include <vector>

#include "method_main.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/cg.hpp"
#include "ssolve/preconditioner.hpp"
#include "ssolve/reorder.hpp"

using namespace ssolve;

int main(int argc, char** argv) {
  return ssolve_methods::method_main(
      argc, argv, "cg_ic0_rcm", "CG + IC(0), CSR, RCM order",
      [](const CsrMatrix& a, const double* b, double* x, const SolveOptions& options) {
        const std::size_t n = static_cast<std::size_t>(a.n_rows);
        const double setup_start = wall_seconds();
        const std::vector<Index> order = reverse_cuthill_mckee(a);
        const CsrMatrix reordered = permute_symmetric(a, order);
        const auto backend = make_seq_backend(clone(reordered));
        const auto precond = make_preconditioner(Precond::IncompleteCholeskyZero, reordered);
        const double setup_seconds = wall_seconds() - setup_start;

        SolveResult result;
        {
          const ssolve_methods::ProfiledRegion region;
          const double solve_start = wall_seconds();
          AlignedArray<double> new_b(n), new_x(n);
          to_new_order(order, b, new_b.data());
          result = cg(*backend, *precond, new_b.data(), new_x.data(), options);
          to_original_order(order, new_x.data(), x);
          result.solve_seconds = wall_seconds() - solve_start;
        }
        result.setup_seconds = setup_seconds;
        verify_result(a, b, x, options, result);
        return result;
      });
}
