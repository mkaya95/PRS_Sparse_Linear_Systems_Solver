// Our solver with other storage formats, orderings and preconditioners, beside the ladder's shipping CSR
// solvers. Each times reordering, storage and preconditioner construction as setup, as run_backend does.
#include <memory>
#include <stdexcept>
#include <vector>

#include "registry.hpp"
#include "ssolve/aligned_array.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/cg.hpp"
#include "ssolve/preconditioner.hpp"
#include "ssolve/reorder.hpp"
#include "ssolve/timer.hpp"
#include "ssolve/verify.hpp"

using namespace ssolve;

namespace ssolve_bench {

namespace {

enum class Format { Csr, Sell };
enum class Order { Natural, Rcm };

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

// Any format with any preconditioner, in the natural order. The backend comes first, so a non-square
// matrix gets its message.
template <Format F, Precond P>
SolveResult run_precond(const CsrMatrix& a, const double* b, double* x, const SolveOptions& options) {
  const double setup_start = wall_seconds();
  const std::unique_ptr<Backend> backend = F == Format::Sell ? make_sell_backend(a) : make_seq_backend(clone(a));
  const std::unique_ptr<Preconditioner> precond = make_preconditioner(P, a);
  const double setup_seconds = wall_seconds() - setup_start;
  SolveResult result = verified_cg(a, *backend, *precond, b, x, options);
  result.setup_seconds = setup_seconds;
  return result;
}

// The system reordered by reverse Cuthill-McKee: P A Pᵀ (P x) = P b. The backend and the preconditioner
// see only the reordered matrix. b goes in and x comes back in the caller's order, inside the solve time,
// and the answer is checked against the matrix the caller gave.
template <Format F, Precond P>
SolveResult run_rcm(const CsrMatrix& a, const double* b, double* x, const SolveOptions& options) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("the solver backend needs a square matrix");
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const double setup_start = wall_seconds();
  const std::vector<Index> order = reverse_cuthill_mckee(a);
  const CsrMatrix reordered = permute_symmetric(a, order);
  const std::unique_ptr<Backend> backend =
      F == Format::Sell ? make_sell_backend(reordered) : make_seq_backend(clone(reordered));
  const std::unique_ptr<Preconditioner> precond = make_preconditioner(P, reordered);
  const double setup_seconds = wall_seconds() - setup_start;

  const double solve_start = wall_seconds();
  AlignedArray<double> new_b(n), new_x(n);
  to_new_order(order, b, new_b.data());
  if (options.use_initial_guess) to_new_order(order, x, new_x.data());
  SolveResult result = cg(*backend, *precond, new_b.data(), new_x.data(), options);
  to_original_order(order, new_x.data(), x);
  result.solve_seconds = wall_seconds() - solve_start;
  result.setup_seconds = setup_seconds;
  verify_result(a, b, x, options, result);
  return result;
}

}  // namespace

std::vector<NamedSolver> variant_solvers() {
  using P = Precond;
  return {
      {"ours-sell", P::None, run_sell<P::None>},
      {"ours-sell-jacobi", P::Jacobi, run_sell<P::Jacobi>},
      {"ours-sgs", P::SymGaussSeidel, run_precond<Format::Csr, P::SymGaussSeidel>},
      {"ours-ic", P::IncompleteCholesky, run_precond<Format::Csr, P::IncompleteCholesky>},
      {"ours-ic-fixed", P::IncompleteCholeskyFixed, run_precond<Format::Csr, P::IncompleteCholeskyFixed>},
      {"ours-ic0", P::IncompleteCholeskyZero, run_precond<Format::Csr, P::IncompleteCholeskyZero>},
      // SELL-C-σ storage with the preconditioners that store z.
      {"ours-sell-sgs", P::SymGaussSeidel, run_precond<Format::Sell, P::SymGaussSeidel>},
      {"ours-sell-ic", P::IncompleteCholesky, run_precond<Format::Sell, P::IncompleteCholesky>},
      {"ours-sell-ic0", P::IncompleteCholeskyZero, run_precond<Format::Sell, P::IncompleteCholeskyZero>},
      // Reverse Cuthill-McKee ordering.
      {"ours-rcm-jacobi", P::Jacobi, run_rcm<Format::Csr, P::Jacobi>},
      {"ours-rcm-ic0", P::IncompleteCholeskyZero, run_rcm<Format::Csr, P::IncompleteCholeskyZero>},
      {"ours-sell-rcm-jacobi", P::Jacobi, run_rcm<Format::Sell, P::Jacobi>},
      {"ours-sell-rcm-ic0", P::IncompleteCholeskyZero, run_rcm<Format::Sell, P::IncompleteCholeskyZero>},
  };
}

}  // namespace ssolve_bench
