// Times named solvers on one matrix and appends one CSV row per repeat.
//
// Usage: bench_cg --matrix SPEC --solver NAME[,NAME...] [--tol 1e-8] [--maxit N] [--repeat 10]
//                 [--warmup 1] [--mode solve|fixed] [--problem random|a1|unit] [--power TEXT]
//                 [--csv FILE] [--label TEXT] [--progress 0|1] [--solutions FILE]
//
// Solvers take turns round by round (A, B, then B, A, ...), so a machine that slows down
// mid-run affects them equally and none always goes first. A solver that fails is reported
// and dropped, the others continue, and the exit status is then 1.
//
// --progress 1 names each stage on stderr ("progress: solve ours round 1 of 2") before it starts,
// outside every timed region, so a viewer such as the browser demo can follow along.
//
// --solutions FILE writes, after the last round, how far each solver's answer is from every other
// solver's and from the true answer (when the problem has a known one), as block pictures for the
// browser page. Only solvers that ran every round are compared.
//
// The problem sets the right-hand side: b = A·x for a fixed random x (the default), b = A·1, or b = 1.
// The rows of A·1 can nearly cancel, which puts the tolerance out of reach on some real matrices.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <map>
#include <set>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "cli.hpp"
#include "ssolve/compare.hpp"
#include "problem.hpp"
#include "ssolve/matrix_spec.hpp"
#include "registry.hpp"
#include "ssolve/aligned_array.hpp"
#include "ssolve/csv.hpp"
#include "ssolve/generators.hpp"
#include "ssolve/manifest.hpp"
#include "ssolve/timer.hpp"
#include "ssolve/verify.hpp"

using namespace ssolve;
using namespace ssolve_bench;

namespace {

// Every pair of solvers in the order they were named, then each solver against the true answer. A run in
// which no solver finished still gets its file, with no comparisons, so an older picture cannot stand in.
void write_solutions(const std::string& path, const std::string& matrix, const std::string& problem, std::size_t n,
                     const std::vector<std::string>& order, const std::map<std::string, std::vector<double>>& answers) {
  std::vector<const std::string*> present;
  for (const std::string& name : order) {
    if (answers.count(name) != 0) present.push_back(&name);
  }
  std::vector<double> truth;
  if (problem == "random") {
    const AlignedArray<double> x = random_solution(static_cast<Index>(n));
    truth.assign(x.begin(), x.end());
  } else if (problem == "a1") {
    truth.assign(n, 1.0);
  }

  std::vector<VectorDifference> diffs;
  for (std::size_t i = 0; i < present.size(); ++i) {
    for (std::size_t j = i + 1; j < present.size(); ++j) {
      diffs.push_back(compare_vectors(*present[i], answers.at(*present[i]).data(), *present[j],
                                      answers.at(*present[j]).data(), n));
    }
  }
  double scale = 0.0;
  if (!truth.empty()) {
    for (const std::string* name : present) {
      diffs.push_back(compare_vectors(*name, answers.at(*name).data(), "true", truth.data(), n));
    }
    for (const double value : truth) scale = std::fmax(scale, std::fabs(value));
  } else {
    for (const std::string* name : present) {
      for (const double value : answers.at(*name)) {
        if (std::isfinite(value)) scale = std::fmax(scale, std::fabs(value));
      }
    }
  }
  write_file_atomically(path, solutions_json(matrix, static_cast<std::int64_t>(n), problem, scale, diffs));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const ssolve_cli::Args args(argc, argv, {"matrix", "solver", "tol", "maxit", "repeat", "warmup", "mode", "problem",
                                             "power", "csv", "label", "progress", "solutions"});
    const std::string spec = args.required("matrix");
    const std::string solver_list = args.required("solver");
    const int repeats = args.get_count("repeat", 10, 1);
    const int warmups = args.get_count("warmup", 1);
    const int progress = args.get_count("progress", 0);
    if (progress > 1) throw std::runtime_error("--progress must be 0 or 1");
    const auto say = [progress](const std::string& stage) {
      if (progress != 0) std::fprintf(stderr, "progress: %s\n", stage.c_str());
    };
    const std::string problem = args.get("problem", "random");
    const std::string mode = args.get("mode", "solve");
    if (mode != "solve" && mode != "fixed") throw std::runtime_error("--mode must be solve or fixed");

    SolveOptions options;
    options.tol = args.get_double("tol", 1e-8);
    options.max_iterations = args.get_int("maxit", 0);
    validate_options(options);
    if (mode == "fixed" && options.max_iterations == 0) throw std::runtime_error("fixed mode needs --maxit");

    say("read");
    const double preparation_start = wall_seconds();
    const CsrMatrix a = load_matrix_spec(spec);
    validate_cg_matrix(a);
    const std::string fingerprint = matrix_fingerprint(a);
    const AlignedArray<double> b = ssolve_cli::rhs_for_problem(problem, a);
    AlignedArray<double> x(static_cast<std::size_t>(a.n_rows));
    const double preparation_seconds = wall_seconds() - preparation_start;
    const std::string run_id = std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "-" +
                               std::to_string(getpid());

    std::vector<std::string> columns = manifest_columns();
    for (const char* column :
         {"matrix", "n", "nnz", "solver", "precond", "tol", "maxit", "repeat", "status", "spmv_count",
          "rel_res_recursive", "rel_res_true", "setup_s", "solve_s", "per_spmv_s", "label",
          "mode", "warmup", "repeats", "preparation_s", "verification_s", "end_to_end_s", "eigen_version",
          "input", "matrix_id", "run_id", "problem", "power"}) {
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
    std::vector<std::string> named_order;  // as given on --solver, for the comparison order
    std::map<std::string, std::vector<double>> answers;  // last measured x per solver, for --solutions
    std::set<std::string> failed;  // solvers with a failed measured solve: never drawn, whatever later rounds do
    std::stringstream names(solver_list);
    for (std::string solver_name; std::getline(names, solver_name, ',');) {
      try {
        solvers.push_back(find_solver(solver_name));
        named_order.push_back(solver_name);
      } catch (const std::exception& e) {
        report_failure(solver_name, e.what());
      }
    }

    const Manifest manifest = collect_manifest();
    const std::string name = matrix_display_name(spec);
    const auto run_once = [&](const NamedSolver& solver, int repeat) {
      // Identical initial state for warm-ups and all implementations.
      x.fill(0.0);
      const double start = wall_seconds();
      const SolveResult result = solver.run(a, b.data(), x.data(), options);
      const double end_to_end = preparation_seconds + wall_seconds() - start;
      if (repeat >= 0 && args.has("solutions") && failed.count(solver.name) == 0) {
        answers[solver.name].assign(x.begin(), x.end());
      }
      if (repeat < 0) {
        if (result.status == SolveStatus::Breakdown || result.status == SolveStatus::ResidualFailure) {
          throw std::runtime_error(std::string(to_string(result.status)) + ": " + result.detail);
        }
        return;
      }
      const double true_residual = result.true_residual;
      const double per_spmv = result.spmv_count > 0 ? result.solve_seconds / static_cast<double>(result.spmv_count) : 0.0;

      std::printf("%s %s n=%d nnz=%lld %s spmv=%lld res=%.3e true=%.3e solve=%.6fs per_iter=%.3eus\n", name.c_str(),
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
                                                 args.get("label", ""),
                                                 mode, std::to_string(warmups), std::to_string(repeats),
                                                 fmt_double(preparation_seconds),
                                                 fmt_double(result.verification_seconds), fmt_double(end_to_end),
                                                 eigen_version(), spec, fingerprint, run_id, problem,
                                                 args.get("power", "unknown")};
        row.insert(row.end(), values.begin(), values.end());
        csv->write_row(row);
      }
      if ((mode == "solve" && result.status != SolveStatus::Converged) ||
          result.status == SolveStatus::Breakdown || result.status == SolveStatus::ResidualFailure) {
        report_failure(solver.name, std::string(to_string(result.status)) + ": " + result.detail);
        // A failed answer is not drawn: the picture would look like a valid answer.
        failed.insert(solver.name);
        answers.erase(solver.name);
      }
    };

    if (solvers.empty()) throw std::runtime_error("no valid solvers selected");
    for (std::int64_t round = -static_cast<std::int64_t>(warmups); round < repeats; ++round) {
      const int repeat = static_cast<int>(round);
      // Rotate who goes first, because the first run in a round is measurably slower.
      if (repeat > 0 && !solvers.empty()) std::rotate(solvers.begin(), solvers.begin() + 1, solvers.end());
      for (auto solver = solvers.begin(); solver != solvers.end();) {
        say("solve " + solver->name +
            (repeat < 0 ? std::string(" warmup")
                        : " round " + std::to_string(repeat + 1) + " of " + std::to_string(repeats)));
        try {
          run_once(*solver, repeat);
          ++solver;
        } catch (const std::exception& e) {
          report_failure(solver->name, e.what());
          answers.erase(solver->name);
          solver = solvers.erase(solver);
        }
      }
    }
    if (args.has("solutions")) {
      write_solutions(args.required("solutions"), name, problem, static_cast<std::size_t>(a.n_rows), named_order,
                      answers);
    }
    return exit_status;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_cg: %s\n", e.what());
    return 1;
  }
}
