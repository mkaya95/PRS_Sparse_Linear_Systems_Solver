// Times each CG kernel for several implementations, so a gap can be traced to its source.
//
// Usage: bench_kernels --matrix SPEC --impl eigen,L0,... [--repeat 10] [--min-ms 50] [--csv FILE]
//
// Implementations take turns, kernel by kernel, so a machine that slows down mid-run
// affects them equally. SpMV swaps its input and output vectors between calls (Asudeh
// et al. 2025), because a fixed x stays in cache in a way it never does inside CG. The
// output is rescaled outside the timed region every 8 calls, so values stay finite.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cli.hpp"
#include "kernels.hpp"
#include "matrix_spec.hpp"
#include "ssolve/aligned_array.hpp"
#include "ssolve/csv.hpp"
#include "ssolve/manifest.hpp"
#include "ssolve/timer.hpp"

using namespace ssolve;
using namespace ssolve_bench;

namespace {

struct Implementation {
  std::string name;
  KernelSet kernels;
};

struct Vectors {
  AlignedArray<double> x, y;
  explicit Vectors(std::size_t n) : x(n), y(n) {
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = 1.0 + static_cast<double>(i % 7) / 7.0;
      y[i] = 2.0 - static_cast<double>(i % 5) / 5.0;
    }
  }
};

void rescale(double* v, std::size_t n) {
  double largest = 0.0;
  for (std::size_t i = 0; i < n; ++i) largest = std::fmax(largest, std::fabs(v[i]));
  if (largest == 0.0) return;
  for (std::size_t i = 0; i < n; ++i) v[i] /= largest;
}

// Seconds per call, timing only the kernel and repeating until min_seconds have been timed.
template <class Call>
double seconds_per_call(double min_seconds, Call call) {
  double timed = 0.0;
  long long calls = 0;
  do {
    const double start = wall_seconds();
    call(calls);
    timed += wall_seconds() - start;
    ++calls;
  } while (timed < min_seconds);
  return timed / static_cast<double>(calls);
}

volatile double sink;  // keeps the compiler from removing kernels whose result is unused

double measure(const std::string& kernel, const KernelSet& k, Vectors& v, std::size_t n, double min_seconds) {
  double* x = v.x.data();
  double* y = v.y.data();
  if (kernel == "spmv") {
    long long calls = 0;
    double timed = 0.0;
    do {
      const double start = wall_seconds();
      k.spmv(x, y);
      timed += wall_seconds() - start;
      std::swap(x, y);
      ++calls;
      if (calls % 8 == 0) rescale(x, n);
    } while (timed < min_seconds);
    return timed / static_cast<double>(calls);
  }
  if (kernel == "dot") {
    return seconds_per_call(min_seconds, [&](long long) { sink = k.dot(x, y); });
  }
  if (kernel == "axpy") {
    return seconds_per_call(min_seconds, [&](long long call) { k.axpy(call % 2 == 0 ? 1e-3 : -1e-3, x, y); });
  }
  if (kernel == "norm2") {
    return seconds_per_call(min_seconds, [&](long long) { sink = k.norm2(x); });
  }
  if (kernel == "xpby") {
    return seconds_per_call(min_seconds, [&](long long) { k.xpby(x, 0.5, y); });
  }
  throw std::logic_error("unknown kernel " + kernel);
}

// Bytes each kernel must move at minimum, the same for every implementation.
double kernel_bytes(const std::string& kernel, double n, double nnz) {
  if (kernel == "spmv") return 12.0 * nnz + 16.0 * n;
  if (kernel == "norm2") return 8.0 * n;
  return 16.0 * n;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const ssolve_cli::Args args(argc, argv, {"matrix", "impl", "repeat", "min-ms", "csv"});
    const std::string spec = args.required("matrix");
    const int repeats = static_cast<int>(args.get_int("repeat", 10));
    const double min_seconds = args.get_double("min-ms", 50.0) / 1000.0;
    if (repeats < 1 || !(min_seconds > 0.0)) throw std::runtime_error("--repeat and --min-ms must be positive");

    const CsrMatrix a = load_matrix_spec(spec);
    if (a.n_rows != a.n_cols) throw std::runtime_error("bench_kernels needs a square matrix: " + spec);
    const std::size_t n = static_cast<std::size_t>(a.n_rows);

    std::vector<Implementation> implementations;
    std::stringstream names(args.required("impl"));
    for (std::string name; std::getline(names, name, ',');) {
      implementations.push_back({name, name == "eigen" ? eigen_kernels(a) : ladder_kernels(name, a)});
    }

    std::vector<std::string> columns = manifest_columns();
    for (const char* column : {"matrix", "n", "nnz", "impl", "kernel", "repeat", "seconds", "gb_per_s"}) {
      columns.push_back(column);
    }
    std::unique_ptr<CsvWriter> csv;
    if (args.has("csv")) csv = std::make_unique<CsvWriter>(args.required("csv"), columns);

    const Manifest manifest = collect_manifest();
    const std::string matrix_name = matrix_display_name(spec);
    Vectors vectors(n);
    for (int repeat = 0; repeat < repeats; ++repeat) {
      for (const char* kernel : {"spmv", "dot", "axpy", "norm2", "xpby"}) {
        for (const Implementation& impl : implementations) {
          const double seconds = measure(kernel, impl.kernels, vectors, n, min_seconds);
          const double gb_per_s = kernel_bytes(kernel, static_cast<double>(n), static_cast<double>(a.nnz())) / 1e9 / seconds;
          std::printf("%s %s %s %.3e s %.2f GB/s\n", matrix_name.c_str(), impl.name.c_str(), kernel, seconds, gb_per_s);
          if (csv) {
            std::vector<std::string> row = manifest_values(manifest);
            const std::vector<std::string> values = {matrix_name,      std::to_string(n), std::to_string(a.nnz()),
                                                     impl.name,        kernel,            std::to_string(repeat),
                                                     fmt_double(seconds), fmt_double(gb_per_s)};
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
