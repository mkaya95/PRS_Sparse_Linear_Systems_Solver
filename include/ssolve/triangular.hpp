// A preconditioner of the form M = K Kᵀ, K lower triangular, applied without branches in its inner loops.
//
// Symmetric Gauss-Seidel, IC(0) and both incomplete Cholesky variants are all of this form:
//   SGS                M = (D + L) D⁻¹ (D + U)    K = (D + L) D^(-1/2)
//   IC(0), Eigen's IC  M⁻¹ = S L⁻ᵀ L⁻¹ S          K = S⁻¹ L
// so one pair of triangular sweeps serves them all. What makes the sweeps fast:
//
// 1. No data-dependent branches. The rows are cut into blocks of kTriangularBlock consecutive rows, and
//    every row of a block is padded to the block's widest row, so a block runs a loop of fixed length,
//    unrolled by the compiler. Padding reads a zero coefficient at a slot that always holds 0.
//    In the plain CSR loops the processor guessed each short row's end wrong, a third of all slots.
// 2. The neighbour stays in a register. Entry (i, i - 1) (in a third of thermal2's rows, half after RCM) is
//    stored apart, and the value just computed is used from a register instead of a store and a reload.
// 3. Both sweeps read in their own direction. The upper triangle Kᵀ is stored by rows too, so the backward
//    sweep gathers like the forward one, instead of scattering updates into z.
// 4. Fewer passes over memory. The CG update x += a p, r -= a q, r·r runs inside the forward sweep, and
//    r·z inside the backward one, so those vectors are not read again in separate loops. The scaling S is
//    folded into K, so there are no scaling passes.
// 5. Half the bytes per coefficient. K is stored in single precision when every coefficient is within
//    float's normal range, and in double otherwise; the sums are done in double either way. A
//    preconditioner only has to approximate A⁻¹, and CG still multiplies by A in double precision and
//    checks the true residual. IC(0) and SGS took as many iterations as the original double-precision
//    code on thermal2, apache2, tmt_sym, ecology2 and G3_circuit.
//
// The preconditioner runs the CG update itself (point 4), on the host, so it does not go through the
// Backend's cg_update. That suits these sweeps, which are sequential by nature: each row needs the rows
// before it. A parallel backend will need its own triangular sweeps (level scheduling or colouring).
#pragma once

#include <cstdint>
#include <memory>

#include "ssolve/aligned_array.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/incomplete_cholesky.hpp"
#include "ssolve/ic0.hpp"
#include "ssolve/matrix.hpp"
#include "ssolve/preconditioner.hpp"

namespace ssolve {

constexpr Index kTriangularBlock = 8;  // rows sharing one padded width
constexpr int kTriangularMaxWidth = 8;  // wider blocks run the general loop, with a branch per row

// One triangle by rows, without the diagonal and without the neighbour entry, in padded blocks.
// Entry k of row i is at block_start[b] + (i - b * kTriangularBlock) * width[b] + k, for b = i / kTriangularBlock.
template <class T>
struct PaddedTriangleT {
  AlignedArray<std::int64_t> block_start;  // blocks + 1
  AlignedArray<Index> width;               // blocks
  AlignedArray<Index> col;                 // padding points at n, the slot that is always 0
  AlignedArray<T> val;                     // padding is 0
};

template <class T>
struct TriangularFactorT {
  Index n = 0;
  PaddedTriangleT<T> lower;    // K below the diagonal, without (i, i - 1)
  PaddedTriangleT<T> upper;    // Kᵀ above the diagonal, without (i, i + 1)
  AlignedArray<T> near;        // near[i] = K(i, i - 1), 0 if absent; near[0] = near[n] = 0
  AlignedArray<T> inv_diag;    // 1 / K(i, i)
};

// Auto stores K in single precision if every coefficient fits (point 5); Double always uses double.
enum class FactorPrecision { Auto, Double };

struct TriangularFactor {
  Index n = 0;
  bool single = false;  // K is in f32 if set, else in f64
  TriangularFactorT<float> f32;
  TriangularFactorT<double> f64;
};

// K from its strict lower triangle by rows (any column order) and its diagonal, taken as given: the callers
// check what their preconditioner requires, and a NaN in K, as Eigen's factor can hold, reaches CG as a
// Breakdown, as before. Throws on mismatched sizes or an entry on or above the diagonal.
TriangularFactor make_triangular_factor(const CsrMatrix& strict_lower, const AlignedArray<double>& diagonal,
                                        FactorPrecision precision = FactorPrecision::Auto);

// The factor of each preconditioner. Throws as the preconditioner's own constructor does.
TriangularFactor sgs_triangular_factor(const CsrMatrix& a, FactorPrecision precision = FactorPrecision::Auto);
TriangularFactor ic0_triangular_factor(const Ic0Factor& f, FactorPrecision precision = FactorPrecision::Auto);
TriangularFactor ic_triangular_factor(const CholeskyFactor& f, FactorPrecision precision = FactorPrecision::Auto);

// w = M⁻¹ r, returning r·w. w holds n + 1 entries, and w[n] must be 0 (the padding slot); it stays 0.
double triangular_apply(const TriangularFactor& f, const double* r, double* w);

// x += a p and r −= a q, then w = M⁻¹ r for the new r, in two sweeps. Returns {r·r, r·w}.
RrRz triangular_update_apply(const TriangularFactor& f, double a, const double* p, const double* q, double* x,
                             double* r, double* w);

std::unique_ptr<Preconditioner> make_triangular_preconditioner(TriangularFactor f);

}  // namespace ssolve
