// Times named solvers on one matrix and appends one CSV row per repeat.
//
// Usage: bench_cg --matrix SPEC --solver NAME[,NAME...] [--tol 1e-8] [--maxit N] [--repeat 10]
//                 [--csv FILE] [--label TEXT]
//
// Solvers take turns round by round (A, B, then B, A, ...), so a machine that slows down
// mid-run affects them equally and none always goes first. A solver that fails is reported
// and dropped, the others continue, and the exit status is then 1.
//
// The right-hand side is b = A·1, so the exact solution is the vector of ones.
#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "ssolve/matrix_spec.hpp"
#include "registry.hpp"
#include "ssolve/aligned_array.hpp"
#include "ssolve/csv.hpp"
#include "ssolve/generators.hpp"
#include "ssolve/manifest.hpp"
#include "ssolve/verify.hpp"

using namespace ssolve;
using namespace ssolve_bench;

int main(int argc, char** argv) {
  try {
    const ssolve_cli::Args args(argc, argv, {"matrix", "solver", "tol", "maxit", "repeat", "csv", "label"});
    const std::string spec = args.required("matrix");
    const std::string solver_list = args.required("solver");
    const int repeats = static_cast<int>(args.get_int("repeat", 10));
    if (repeats < 1) throw std::runtime_error("--repeat must be positive");

    SolveOptions options;
    options.tol = args.get_double("tol", 1e-8);
    options.max_iterations = args.get_int("maxit", 0);

    const CsrMatrix a = load_matrix_spec(spec);
    if (a.n_rows != a.n_cols) throw std::runtime_error("CG needs a square matrix: " + spec);
    const double defect = symmetry_defect(a);
    if (defect > 1e-8) {
      throw std::runtime_error("matrix is not symmetric (defect " + fmt_double(defect) + "), so CG does not apply: " + spec);
    }
    const AlignedArray<double> b = rhs_for_unit_solution(a);
    AlignedArray<double> x(static_cast<std::size_t>(a.n_rows));

    std::vector<std::string> columns = manifest_columns();
    for (const char* column :
         {"matrix", "n", "nnz", "solver", "precond", "tol", "maxit", "repeat", "status", "spmv_count",
          "rel_res_recursive", "rel_res_true", "setup_s", "solve_s", "per_spmv_s", "label"}) {
      columns.push_back(column);
    }
    std::unique_ptr<CsvWriter> csv;
    if (args.has("csv")) csv = std::make_unique<CsvWriter>(args.required("csv"), columns);

    int exit_status = 0;
    const auto report_failure = [&](const std::string& solver_name, const std::string& message) {
      std::fprintf(stderr, "bench_cg: %s: %s\n", solver_name.c_str(), message.c_str());
      exit_status = 1;
    };

    std::vector<NamedSolver> solvers;
    std::stringstream names(solver_list);
    for (std::string solver_name; std::getline(names, solver_name, ',');) {
      try {
        solvers.push_back(find_solver(solver_name));
      } catch (const std::exception& e) {
        report_failure(solver_name, e.what());
      }
    }

    const Manifest manifest = collect_manifest();
    const std::string name = matrix_display_name(spec);
    const auto run_once = [&](const NamedSolver& solver, int repeat) {
      const SolveResult result = solver.run(a, b.data(), x.data(), options);
      const double true_residual = true_relative_residual(a, b.data(), x.data());
      const double per_spmv = result.spmv_count > 0 ? result.solve_seconds / static_cast<double>(result.spmv_count) : 0.0;

      std::printf("%s %s n=%d nnz=%lld %s spmv=%lld res=%.3e true=%.3e solve=%.6fs per_spmv=%.3eus\n", name.c_str(),
                  solver.name.c_str(), a.n_rows, static_cast<long long>(a.nnz()), to_string(result.status),
                  static_cast<long long>(result.spmv_count), result.recursive_residual, true_residual,
                  result.solve_seconds, per_spmv * 1e6);
      std::fflush(stdout);

      if (csv) {
        std::vector<std::string> row = manifest_values(manifest);
        const std::vector<std::string> values = {name,
                                                 std::to_string(a.n_rows),
                                                 std::to_string(a.nnz()),
                                                 solver.name,
                                                 to_string(solver.precond),
                                                 fmt_double(options.tol),
                                                 std::to_string(options.max_iterations),
                                                 std::to_string(repeat),
                                                 to_string(result.status),
                                                 std::to_string(result.spmv_count),
                                                 fmt_double(result.recursive_residual),
                                                 fmt_double(true_residual),
                                                 fmt_double(result.setup_seconds),
                                                 fmt_double(result.solve_seconds),
                                                 fmt_double(per_spmv),
                                                 args.get("label", "")};
        row.insert(row.end(), values.begin(), values.end());
        csv->write_row(row);
      }
    };

    for (int repeat = 0; repeat < repeats; ++repeat) {
      // Rotate who goes first, because the first run in a round is measurably slower.
      if (repeat > 0 && !solvers.empty()) std::rotate(solvers.begin(), solvers.begin() + 1, solvers.end());
      for (auto solver = solvers.begin(); solver != solvers.end();) {
        try {
          run_once(*solver, repeat);
          ++solver;
        } catch (const std::exception& e) {
          report_failure(solver->name, e.what());
          solver = solvers.erase(solver);
        }
      }
    }
    return exit_status;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_cg: %s\n", e.what());
    return 1;
  }
}
