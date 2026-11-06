// Times each CG kernel for several implementations, so a gap can be traced to its source.
//
// Usage: bench_kernels --matrix SPEC --impl eigen,L0,... [--warmup 1] [--repeat 10] [--min-ms 50] [--csv FILE]
//
// Implementations take turns, kernel by kernel, and rotate who goes first each round, so
// a machine that slows down mid-run affects them equally. SpMV swaps its input and output vectors between calls (Asudeh
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
#include "ssolve/matrix_spec.hpp"
#include "ssolve/aligned_array.hpp"
#include "ssolve/csv.hpp"
#include "ssolve/manifest.hpp"
#include "ssolve/timer.hpp"
#include "ssolve/verify.hpp"

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
    reset();
  }
  void reset() {
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = 1.0 + static_cast<double>(i % 7) / 7.0;
      y[i] = 2.0 - static_cast<double>(i % 5) / 5.0;
    }
  }
};

void rescale(double* v, std::size_t n) {
  double largest = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(v[i])) throw std::runtime_error("non-finite kernel output");
    largest = std::fmax(largest, std::fabs(v[i]));
  }
  if (largest == 0.0) return;
  for (std::size_t i = 0; i < n; ++i) v[i] /= largest;
}

struct Sample {
  double seconds_per_call;
  long long calls;
};

// Times only `call`, until min_seconds have been timed and at least kMinCalls calls were made.
// `between` runs untimed after each call.
constexpr long long kMinCalls = 4;

template <class Call, class Between>
Sample timed_calls(double min_seconds, Call call, Between between) {
  double timed = 0.0;
  long long calls = 0;
  do {
    const double start = wall_seconds();
    call(calls);
    timed += wall_seconds() - start;
    ++calls;
    between(calls);
  } while (timed < min_seconds || calls < kMinCalls);
  return {timed / static_cast<double>(calls), calls};
}

template <class Call>
Sample timed_calls(double min_seconds, Call call) {
  return timed_calls(min_seconds, call, [](long long) {});
}

volatile double sink;  // keeps the compiler from removing kernels whose result is unused

Sample measure(const std::string& kernel, const KernelSet& k, Vectors& v, std::size_t n, double min_seconds) {
  double* x = v.x.data();
  double* y = v.y.data();
  if (kernel == "spmv") {
    return timed_calls(
        min_seconds, [&](long long) { k.spmv(x, y); },
        [&](long long calls) {
          std::swap(x, y);
          if (calls % 8 == 0) rescale(x, n);
        });
  }
  if (kernel == "dot") {
    return timed_calls(min_seconds, [&](long long) { sink = k.dot(x, y); });
  }
  if (kernel == "axpy") {
    return timed_calls(min_seconds, [&](long long call) { k.axpy(call % 2 == 0 ? 1e-3 : -1e-3, x, y); });
  }
  if (kernel == "norm2") {
    return timed_calls(min_seconds, [&](long long) { sink = k.norm2(x); });
  }
  if (kernel == "xpby") {
    return timed_calls(min_seconds, [&](long long) { k.xpby(x, 0.5, y); });
  }
  throw std::logic_error("unknown kernel " + kernel);
}

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
          vectors.reset();
          const Sample sample = measure(kernel, impl.kernels, vectors, n, min_seconds);
          for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(vectors.x[i]) || !std::isfinite(vectors.y[i])) {
              throw std::runtime_error("non-finite kernel output");
            }
          }
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
