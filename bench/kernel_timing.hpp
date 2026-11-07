// Timing loops behind bench_kernels.
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "kernels.hpp"
#include "ssolve/aligned_array.hpp"
#include "ssolve/timer.hpp"

namespace ssolve_bench {

struct Vectors {
  ssolve::AlignedArray<double> x, y;
  explicit Vectors(std::size_t n) : x(n), y(n) { reset(); }

  void reset() {
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = 1.0 + static_cast<double>(i % 7) / 7.0;
      y[i] = 2.0 - static_cast<double>(i % 5) / 5.0;
    }
  }
};

inline void rescale(double* v, std::size_t n) {
  double largest = 0.0;
  for (std::size_t i = 0; i < n; ++i) largest = std::fmax(largest, std::fabs(v[i]));
  if (largest == 0.0 || !std::isfinite(largest)) return;
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
    const double start = ssolve::wall_seconds();
    call(calls);
    timed += ssolve::wall_seconds() - start;
    ++calls;
    between(calls);
  } while (timed < min_seconds || calls < kMinCalls);
  return {timed / static_cast<double>(calls), calls};
}

inline volatile double kernel_sink;  // keeps the compiler from removing kernels whose result is unused

// One timed sample of a kernel. Every sample starts from the same vectors, because repeated SpMV
// acts as a power iteration and would otherwise spread values over a range that never occurs in CG.
inline Sample measure_kernel(const std::string& kernel, const KernelSet& k, Vectors& v, double min_seconds) {
  v.reset();
  const std::size_t n = v.x.size();
  double* x = v.x.data();
  double* y = v.y.data();
  const auto nothing = [](long long) {};
  Sample sample{};
  if (kernel == "spmv") {
    sample = timed_calls(
        min_seconds, [&](long long) { k.spmv(x, y); },
        [&](long long calls) {
          std::swap(x, y);
          if (calls % 8 == 0) rescale(x, n);
        });
  } else if (kernel == "dot") {
    sample = timed_calls(min_seconds, [&](long long) { kernel_sink = k.dot(x, y); }, nothing);
  } else if (kernel == "axpy") {
    sample = timed_calls(min_seconds, [&](long long call) { k.axpy(call % 2 == 0 ? 1e-3 : -1e-3, x, y); }, nothing);
  } else if (kernel == "norm2") {
    sample = timed_calls(min_seconds, [&](long long) { kernel_sink = k.norm2(x); }, nothing);
  } else if (kernel == "xpby") {
    sample = timed_calls(min_seconds, [&](long long) { k.xpby(x, 0.5, y); }, nothing);
  } else {
    throw std::logic_error("unknown kernel " + kernel);
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(v.x[i]) || !std::isfinite(v.y[i])) throw std::runtime_error("non-finite kernel output");
  }
  return sample;
}

}  // namespace ssolve_bench
