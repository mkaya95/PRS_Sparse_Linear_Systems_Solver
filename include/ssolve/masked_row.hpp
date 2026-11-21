// One sparse row times a vector with no branch on the row's length, for rows stored back to back (CSR).
//
// A loop over a short row ends after a varying number of turns, and the processor guesses that end
// wrong for a large share of rows (VTune's "bad speculation"). Here every row is processed as W lanes
// (W = 4, 8 or 16) with AVX2: the lanes past the row's end are switched off by a mask, in the loads and
// in the gather from x, not by a branch, so the work per row is the same whatever its length. Only rows
// longer than W take a loop, and that branch is predictable when such rows are rare.
//
// The arrays are still read front to back, as in plain CSR, so no extra bytes come from memory, and
// masked-off lanes read nothing, so no slack is needed past the last row.
//
// Without AVX2 and FMA the plain row loop is used.
#pragma once

#include <cstdint>

#include "ssolve/kernels_common.hpp"
#include "ssolve/matrix.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define SSOLVE_MASKED_AVX2 1
#endif

namespace ssolve {

#ifdef SSOLVE_MASKED_AVX2

namespace masked_detail {

// acc += val[k] x[col[k]] for the 4 lanes k = first .. first + 3 that are below `length`.
inline __m256d lanes4(const Index* col, const double* val, const double* x, std::int64_t first, std::int64_t length,
                      __m256d acc) {
  // Lanes j < length - first are on. A row has fewer than 2^31 entries, so the difference fits in 32 bits.
  const int left = static_cast<int>(length - first);
  const __m128i mask32 = _mm_cmpgt_epi32(_mm_set1_epi32(left), _mm_setr_epi32(0, 1, 2, 3));
  const __m256i mask64 = _mm256_cvtepi32_epi64(mask32);
  const __m128i index = _mm_maskload_epi32(col + first, mask32);
  const __m256d xs = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), x, index, _mm256_castsi256_pd(mask64), 8);
  const __m256d vs = _mm256_maskload_pd(val + first, mask64);
  return _mm256_fmadd_pd(vs, xs, acc);
}

inline double sum4(__m256d v) {
  const __m128d pair = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
  return _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)));
}

}  // namespace masked_detail

// Σ val[k] x[col[k]] for k < length.
template <int W>
inline double masked_row(const Index* SSOLVE_RESTRICT col, const double* SSOLVE_RESTRICT val, const double* x,
                         std::int64_t length) {
  static_assert(W == 4 || W == 8 || W == 16, "W is 4, 8 or 16");
  using masked_detail::lanes4;
  __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd(), a2 = _mm256_setzero_pd(), a3 = _mm256_setzero_pd();
  std::int64_t k = 0;
  do {  // once for rows of at most W entries: the loop branch is the only branch, and rarely taken again
    a0 = lanes4(col, val, x, k, length, a0);
    if (W >= 8) a1 = lanes4(col, val, x, k + 4, length, a1);
    if (W >= 16) {
      a2 = lanes4(col, val, x, k + 8, length, a2);
      a3 = lanes4(col, val, x, k + 12, length, a3);
    }
    k += W;
  } while (k < length);
  if (W >= 16) {
    a0 = _mm256_add_pd(a0, a2);
    a1 = _mm256_add_pd(a1, a3);
  }
  if (W >= 8) a0 = _mm256_add_pd(a0, a1);
  return masked_detail::sum4(a0);
}

#else

template <int W>
inline double masked_row(const Index* SSOLVE_RESTRICT col, const double* SSOLVE_RESTRICT val, const double* x,
                         std::int64_t length) {
  double sum0 = 0.0, sum1 = 0.0;
  std::int64_t k = 0;
  for (; k + 1 < length; k += 2) {
    sum0 += val[k] * x[col[k]];
    sum1 += val[k + 1] * x[col[k + 1]];
  }
  if (k < length) sum0 += val[k] * x[col[k]];
  return sum0 + sum1;
}

#endif

// The W for these rows: 4 or 8 if at least 97% of the rows have at most that many entries, else 16.
template <class Ptr>
inline int masked_width(const Ptr* row_ptr, Index rows) {
  std::int64_t up_to_4 = 0, up_to_8 = 0;
  for (Index i = 0; i < rows; ++i) {
    const std::int64_t length = static_cast<std::int64_t>(row_ptr[i + 1] - row_ptr[i]);
    up_to_4 += length <= 4;
    up_to_8 += length <= 8;
  }
  if (up_to_4 * 100 >= static_cast<std::int64_t>(rows) * 97) return 4;
  if (up_to_8 * 100 >= static_cast<std::int64_t>(rows) * 97) return 8;
  return 16;
}

}  // namespace ssolve
