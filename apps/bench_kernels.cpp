// Times each CG kernel for several implementations, so a gap can be traced to its source.
//
// Usage: bench_kernels --matrix SPEC --impl eigen,L0,... [--warmup 1] [--repeat 10] [--min-ms 50] [--csv FILE]
//
// Implementations take turns, kernel by kernel, and rotate who goes first each round, so
// a machine that slows down mid-run affects them equally. SpMV swaps its input and output vectors between calls (Asudeh
// et al. 2025), because a fixed x stays in cache in a way it never does inside CG. Outside the
// timed region, each sample starts from the same vectors and SpMV output is rescaled every 8 calls.
#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cli.hpp"
#include "kernel_timing.hpp"
#include "kernels.hpp"
#include "ssolve/matrix_spec.hpp"
#include "ssolve/aligned_array.hpp"
#include "ssolve/csv.hpp"
#include "ssolve/manifest.hpp"
#include "ssolve/verify.hpp"

using namespace ssolve;
using namespace ssolve_bench;

namespace {

struct Implementation {
  std::string name;
  KernelSet kernels;
};

// Bytes each kernel must move at minimum, the same for every implementation.
double kernel_bytes(const std::string& kernel, double n, double nnz) {
  if (kernel == "spmv") return 12.0 * nnz + 24.0 * n + 8.0;
  if (kernel == "norm2") return 8.0 * n;
  if (kernel == "axpy" || kernel == "xpby") return 24.0 * n;
  return 16.0 * n;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const ssolve_cli::Args args(argc, argv, {"matrix", "impl", "repeat", "warmup", "min-ms", "csv"});
    const std::string spec = args.required("matrix");
    const int repeats = args.get_count("repeat", 10, 1);
    const int warmups = args.get_count("warmup", 1);
    const double min_seconds = args.get_double("min-ms", 50.0) / 1000.0;
    if (repeats < 1 || !(min_seconds > 0.0)) throw std::runtime_error("--repeat and --min-ms must be positive");

    const CsrMatrix a = load_matrix_spec(spec);
    if (a.n_rows != a.n_cols) throw std::runtime_error("bench_kernels needs a square matrix: " + spec);
    const std::size_t n = static_cast<std::size_t>(a.n_rows);
    const std::string fingerprint = matrix_fingerprint(a);

    std::vector<Implementation> implementations;
    std::stringstream names(args.required("impl"));
    for (std::string name; std::getline(names, name, ',');) {
      implementations.push_back({name, name == "eigen" ? eigen_kernels(a) : ladder_kernels(name, a)});
    }
    if (implementations.empty()) throw std::runtime_error("no implementations selected");

    std::vector<std::string> columns = manifest_columns();
    for (const char* column :
         {"matrix", "n", "nnz", "impl", "kernel", "repeat", "calls", "seconds", "gb_per_s", "warmup", "repeats",
          "matrix_id"}) {
      columns.push_back(column);
    }
    std::unique_ptr<CsvWriter> csv;
    if (args.has("csv")) csv = std::make_unique<CsvWriter>(args.required("csv"), columns);

    const Manifest manifest = collect_manifest();
    const std::string matrix_name = matrix_display_name(spec);
    Vectors vectors(n);
    for (std::int64_t round = -static_cast<std::int64_t>(warmups); round < repeats; ++round) {
      const int repeat = static_cast<int>(round);
      // Rotate who goes first, because the first run in a round is measurably slower.
      if (repeat > 0) std::rotate(implementations.begin(), implementations.begin() + 1, implementations.end());
      for (const char* kernel : {"spmv", "dot", "axpy", "norm2", "xpby"}) {
        for (const Implementation& impl : implementations) {
          const Sample sample = measure_kernel(kernel, impl.kernels, vectors, min_seconds);
          if (repeat < 0) continue;
          const double bytes = kernel_bytes(kernel, static_cast<double>(n), static_cast<double>(a.nnz()));
          const double gb_per_s = bytes / 1e9 / sample.seconds_per_call;
          std::printf("%s %s %s %.3e s %.2f modeled GB/s\n", matrix_name.c_str(), impl.name.c_str(), kernel,
                      sample.seconds_per_call, gb_per_s);
          if (csv) {
            std::vector<std::string> row = manifest_values(manifest);
            const std::vector<std::string> values = {matrix_name,
                                                     std::to_string(n),
                                                     std::to_string(a.nnz()),
                                                     impl.name,
                                                     kernel,
                                                     std::to_string(repeat),
                                                     std::to_string(sample.calls),
                                                     fmt_double(sample.seconds_per_call),
                                                     fmt_double(gb_per_s),
                                                     std::to_string(warmups), std::to_string(repeats), fingerprint};
            row.insert(row.end(), values.begin(), values.end());
            csv->write_row(row);
          }
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_kernels: %s\n", e.what());
    return 1;
  }
}
