// Solves A x = b for one matrix and reports what happened.
//
// Usage: solve --matrix SPEC [--rhs b.txt | --problem a1|random|unit] [--precond none|jacobi] [--tol 1e-8]
//              [--maxit N] [--out x.txt] [--history h.txt]
//
// SPEC is poisson2d:N, poisson3d:N, point27:N, poisson3d:llc=4 or a Matrix Market file.
// Without --rhs, b comes from --problem as in bench_cg, and defaults to b = A·1, a manufactured/demo problem.
// --history writes ‖r‖ / ‖b‖ before the first iteration and after each one, one value per line.
//
// Exit status: 0 converged, 2 not converged or breakdown, 1 input error.
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "problem.hpp"
#include "ssolve/matrix_spec.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/cg.hpp"
#include "ssolve/generators.hpp"
#include "ssolve/timer.hpp"
#include "ssolve/verify.hpp"

using namespace ssolve;

namespace {

template <class Values>
void write_lines(const std::string& path, const Values& values, const char* what) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error(std::string("cannot write ") + what + " file: " + path);
  out.precision(17);
  for (const double value : values) out << value << '\n';
  if (!out) throw std::runtime_error(std::string("failed while writing ") + what + " file: " + path);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const ssolve_cli::Args args(argc, argv, {"matrix", "rhs", "problem", "precond", "tol", "maxit", "out", "history"});
    const std::string spec = args.required("matrix");
    const std::string precond = args.get("precond", "none");
    if (precond != "none" && precond != "jacobi") {
      throw std::runtime_error("unknown preconditioner '" + precond + "': use none or jacobi");
    }
    if (args.has("rhs") && args.has("problem")) throw std::runtime_error("give either --rhs or --problem, not both");

    SolveOptions options;
    options.tol = args.get_double("tol", 1e-8);
    options.max_iterations = args.get_int("maxit", 0);
    validate_options(options);

    const double end_to_end_start = wall_seconds();
    const CsrMatrix a = load_matrix_spec(spec);
    validate_cg_matrix(a);

    const AlignedArray<double> b = args.has("rhs") ? read_rhs(args.required("rhs"), a.n_rows)
                                                   : ssolve_cli::rhs_for_problem(args.get("problem", "a1"), a);
    AlignedArray<double> x(static_cast<std::size_t>(a.n_rows));
    const double preparation_seconds = wall_seconds() - end_to_end_start;

    const double setup_start = wall_seconds();
    AlignedArray<double> inv_diag;
    if (precond == "jacobi") inv_diag = jacobi_inverse_diagonal(a);
    const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
    const double setup_seconds = wall_seconds() - setup_start;

    const double* diag = precond == "jacobi" ? inv_diag.data() : nullptr;
    std::vector<double> history;
    SolveResult result;
    if (args.has("history")) {
      result = cg_with_history(*backend, diag, b.data(), x.data(), options, history);
      verify_result(a, b.data(), x.data(), options, result);
    } else {
      result = verified_cg(a, *backend, diag, b.data(), x.data(), options);
    }
    const double end_to_end_seconds = wall_seconds() - end_to_end_start;

    std::printf("matrix         %s\n", spec.c_str());
    std::printf("size           n = %d, nnz = %lld\n", a.n_rows, static_cast<long long>(a.nnz()));
    std::printf("preconditioner %s\n", precond.c_str());
    std::printf("status         %s\n", to_string(result.status));
    if (result.detail[0] != '\0') std::printf("reason         %s\n", result.detail);
    std::printf("spmv count     %lld\n", static_cast<long long>(result.spmv_count));
    std::printf("setup time     %.6f s\n", setup_seconds);
    std::printf("preparation time %.6f s\n", preparation_seconds);
    std::printf("solve time     %.6f s\n", result.solve_seconds);
    std::printf("recursive residual %.3e\n", result.recursive_residual);
    std::printf("true residual  %.3e\n", result.true_residual);
    std::printf("verification time %.6f s\n", result.verification_seconds);
    std::printf("end-to-end time %.6f s\n", end_to_end_seconds);

    if (args.has("out")) write_lines(args.required("out"), x, "solution");
    if (args.has("history")) write_lines(args.required("history"), history, "history");
    return result.status == SolveStatus::Converged ? 0 : 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "solve: %s\n", e.what());
    return 1;
  }
}
