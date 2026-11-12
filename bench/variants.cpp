// Our solver with other storage formats, beside the ladder's shipping CSR solvers.
// Each times storage construction as setup, as run_backend does for CSR.
#include <memory>

#include "registry.hpp"
#include "ssolve/aligned_array.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/cg.hpp"
#include "ssolve/timer.hpp"

using namespace ssolve;

namespace ssolve_bench {

namespace {

template <Precond P>
SolveResult run_sell(const CsrMatrix& a, const double* b, double* x, const SolveOptions& options) {
  const double setup_start = wall_seconds();
  const std::unique_ptr<Backend> backend = make_sell_backend(a);
  AlignedArray<double> inv_diag;
  if (P == Precond::Jacobi) inv_diag = jacobi_inverse_diagonal(a);
  const double setup_seconds = wall_seconds() - setup_start;
  SolveResult result = verified_cg(a, *backend, P == Precond::Jacobi ? inv_diag.data() : nullptr, b, x, options);
  result.setup_seconds = setup_seconds;
  return result;
}

}  // namespace

std::vector<NamedSolver> variant_solvers() {
  return {
      {"ours-sell", Precond::None, run_sell<Precond::None>},
      {"ours-sell-jacobi", Precond::Jacobi, run_sell<Precond::Jacobi>},
  };
}

}  // namespace ssolve_bench
