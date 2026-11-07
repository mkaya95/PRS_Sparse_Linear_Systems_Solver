// STREAM memory bandwidth benchmark (Copy, Scale, Add, Triad), plus a read-only sum.
//
// Usage: stream [--threads N] [--csv FILE] [--array-mb M] [--repeat 10]
//
// Arrays default to at least 4x the last-level cache, so the numbers describe
// main memory. The threads first-touch their pages with the same static schedule
// as the kernels. SpMV mostly reads, so the read-only rate is its upper bound
// (Kreutzer et al. 2014, Listing 1).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cli.hpp"
#include "ssolve/aligned_array.hpp"
#include "ssolve/csv.hpp"
#include "ssolve/manifest.hpp"
#include "ssolve/timer.hpp"

using namespace ssolve;

namespace {

#ifdef _OPENMP
#define SSOLVE_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#define SSOLVE_PARALLEL_SUM _Pragma("omp parallel for simd schedule(static) reduction(+ : sum)")
#else
#define SSOLVE_PARALLEL_FOR
#define SSOLVE_PARALLEL_SUM _Pragma("omp simd reduction(+ : sum)")
#endif

struct Kernel {
  std::string name;
  int arrays_per_element;  // arrays read plus arrays written
  double best_seconds;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const ssolve_cli::Args args(argc, argv, {"threads", "csv", "array-mb", "repeat"});
    const int threads = static_cast<int>(args.get_int("threads", 1));
    const int repeats = static_cast<int>(args.get_int("repeat", 10));
    if (threads < 1 || repeats < 1) throw std::runtime_error("--threads and --repeat must be positive");
#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    if (threads != 1) throw std::runtime_error("this build has no OpenMP, so only --threads 1 works");
#endif

    const Manifest manifest = collect_manifest();
    const std::int64_t default_bytes = std::max<std::int64_t>(4 * manifest.l3_bytes, 256 * 1024 * 1024);
    const std::int64_t array_bytes = args.has("array-mb") ? args.get_int("array-mb", 0) * 1024 * 1024 : default_bytes;
    const std::int64_t n = array_bytes / static_cast<std::int64_t>(sizeof(double));
    if (n < 1) throw std::runtime_error("--array-mb must be positive");

    AlignedArray<double> a_array(n), b_array(n), c_array(n);
    double* const a = a_array.data();
    double* const b = b_array.data();
    double* const c = c_array.data();
    SSOLVE_PARALLEL_FOR
    for (std::int64_t i = 0; i < n; ++i) {
      a[i] = 1.0;
      b[i] = 2.0;
      c[i] = 0.0;
    }

    const double s = 3.0;
    std::vector<Kernel> kernels = {
        {"read", 1, 1e30}, {"copy", 2, 1e30}, {"scale", 2, 1e30}, {"add", 3, 1e30}, {"triad", 3, 1e30}};
    double last_sum = 0.0;
    for (int r = 0; r < repeats; ++r) {
      double start = wall_seconds();
      double sum = 0.0;
      SSOLVE_PARALLEL_SUM
      for (std::int64_t i = 0; i < n; ++i) sum += a[i];
      kernels[0].best_seconds = std::min(kernels[0].best_seconds, wall_seconds() - start);
      last_sum = sum;

      start = wall_seconds();
      SSOLVE_PARALLEL_FOR
      for (std::int64_t i = 0; i < n; ++i) c[i] = a[i];
      kernels[1].best_seconds = std::min(kernels[1].best_seconds, wall_seconds() - start);

      start = wall_seconds();
      SSOLVE_PARALLEL_FOR
      for (std::int64_t i = 0; i < n; ++i) b[i] = s * c[i];
      kernels[2].best_seconds = std::min(kernels[2].best_seconds, wall_seconds() - start);

      start = wall_seconds();
      SSOLVE_PARALLEL_FOR
      for (std::int64_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
      kernels[3].best_seconds = std::min(kernels[3].best_seconds, wall_seconds() - start);

      start = wall_seconds();
      SSOLVE_PARALLEL_FOR
      for (std::int64_t i = 0; i < n; ++i) a[i] = b[i] + s * c[i];
      kernels[4].best_seconds = std::min(kernels[4].best_seconds, wall_seconds() - start);
    }

    // Standard STREAM validation: after `repeats` rounds the values are known in closed form.
    double expected_a = 1.0, expected_b = 2.0, expected_c = 0.0, expected_sum = 0.0;
    for (int r = 0; r < repeats; ++r) {
      expected_sum = static_cast<double>(n) * expected_a;
      expected_c = expected_a;
      expected_b = s * expected_c;
      expected_c = expected_a + expected_b;
      expected_a = expected_b + s * expected_c;
    }
    if (a[n - 1] != expected_a || b[n - 1] != expected_b || c[n - 1] != expected_c ||
        std::fabs(last_sum - expected_sum) > 1e-8 * expected_sum) {
      throw std::runtime_error("STREAM results failed validation");
    }

    std::vector<std::string> columns = manifest_columns();
    for (const char* column : {"threads", "kernel", "array_bytes", "best_gb_per_s"}) columns.push_back(column);
    std::unique_ptr<CsvWriter> csv;
    if (args.has("csv")) csv = std::make_unique<CsvWriter>(args.required("csv"), columns);

    std::printf("%-8s %12s   (%d thread(s), %.0f MiB arrays)\n", "kernel", "best GB/s", threads,
                static_cast<double>(array_bytes) / (1024.0 * 1024.0));
    for (const Kernel& k : kernels) {
      const double gb_per_s = static_cast<double>(k.arrays_per_element) * static_cast<double>(array_bytes) / 1e9 /
                              k.best_seconds;
      std::printf("%-8s %12.2f\n", k.name.c_str(), gb_per_s);
      if (csv) {
        std::vector<std::string> row = manifest_values(manifest);
        row.push_back(std::to_string(threads));
        row.push_back(k.name);
        row.push_back(std::to_string(array_bytes));
        row.push_back(fmt_double(gb_per_s));
        csv->write_row(row);
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "stream: %s\n", e.what());
    return 1;
  }
}
