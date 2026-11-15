// The sequential SELL-C-σ backend. The vector kernels are the CSR backend's (seq_vectors.hpp).
//
// The SpMV is Listing 4 of Kreutzer et al. 2014 (docs/papers/01_Kreutzer2014_SELL-C-sigma.pdf),
// written there for C = 4 and here for any C as a compile-time constant, so the C rows of a chunk
// become one vector loop. The paper stores y in sorted order; this writes each row's sum to its
// original index instead, so every vector the solver sees stays in the original order.
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include "seq_vectors.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/kernels_common.hpp"
#include "ssolve/sell.hpp"

namespace ssolve {

namespace {

template <int C>
class SellBackend final : public SeqVectorBackend {
 public:
  explicit SellBackend(SellMatrix s) : SeqVectorBackend(static_cast<std::size_t>(s.n_rows)), s_(std::move(s)) {}

  Index n() const override { return s_.n_rows; }

  void spmv(const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y) const override {
    const std::size_t chunks = s_.chunk_width.size();
    for (std::size_t c = 0; c < chunks; ++c) {
      double sum[C];
      chunk_times_vector(c, x, sum);
      const Index* const rows = s_.row_of.data() + c * C;
      for (int r = 0; r < C; ++r) {
        if (rows[r] >= 0) y[rows[r]] = sum[r];
      }
    }
  }

  double spmv_dot(const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y) const override {
    const std::size_t chunks = s_.chunk_width.size();
    double dot = 0.0;
    for (std::size_t c = 0; c < chunks; ++c) {
      double sum[C];
      chunk_times_vector(c, x, sum);
      const Index* const rows = s_.row_of.data() + c * C;
      for (int r = 0; r < C; ++r) {
        if (rows[r] >= 0) {
          y[rows[r]] = sum[r];
          dot += x[rows[r]] * sum[r];
        }
      }
    }
    return dot;
  }

 private:
  // sum[r] = row r of chunk c times x, one column of the chunk at a time (Listing 4's inner loop).
  inline void chunk_times_vector(std::size_t c, const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT sum) const {
    const double* SSOLVE_RESTRICT const val = s_.val.data() + s_.chunk_start[c];
    const Index* SSOLVE_RESTRICT const col = s_.col.data() + s_.chunk_start[c];
    const Index width = s_.chunk_width[c];
    for (int r = 0; r < C; ++r) sum[r] = 0.0;
    for (Index j = 0; j < width; ++j) {
#pragma omp simd
      for (int r = 0; r < C; ++r) sum[r] += val[j * C + r] * x[col[j * C + r]];
    }
  }

  SellMatrix s_;
};

}  // namespace

std::unique_ptr<Backend> make_sell_backend(const CsrMatrix& a, int chunk, Index sigma) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("the solver backend needs a square matrix");
  return make_sell_backend(sell_from_csr(a, chunk, sigma));
}

std::unique_ptr<Backend> make_sell_backend(SellMatrix s) {
  if (s.n_rows != s.n_cols) throw std::runtime_error("the solver backend needs a square matrix");
  switch (s.chunk) {
    case 4:
      return std::make_unique<SellBackend<4>>(std::move(s));
    case 8:
      return std::make_unique<SellBackend<8>>(std::move(s));
    case 16:
      return std::make_unique<SellBackend<16>>(std::move(s));
    case 32:
      return std::make_unique<SellBackend<32>>(std::move(s));
    default:
      throw std::runtime_error("SELL chunk must be 4, 8, 16 or 32");
  }
}

}  // namespace ssolve
