// Jacobi-preconditioned CG run on the scaled system S A S, against the usual PCG with M = D.
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ssolve/backend.hpp"
#include "ssolve/cg.hpp"
#include "ssolve/generators.hpp"
#include "ssolve/jacobi_scaled.hpp"
#include "ssolve/preconditioner.hpp"
#include "ssolve/verify.hpp"
#include "test_util.hpp"

using namespace ssolve;

namespace {

// A with a diagonal raised by a varying factor, so that the scaling is not a multiple of the identity.
CsrMatrix uneven(Stencil stencil, Index side) {
  CsrMatrix a = make_stencil(stencil, side);
  for (Index i = 0; i < a.n_rows; ++i) {
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      if (a.col[k] == i) a.val[k] *= 1.5 + 0.5 * std::sin(0.3 * i);  // 1 to 2 times: still SPD
    }
  }
  return a;
}

}  // namespace

TEST(unit_diagonal_backends_multiply_by_identity_plus_off_diagonal) {
  const CsrMatrix a = uneven(Stencil::Poisson3D, 6);
  const JacobiScaling s = jacobi_scaling(a);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  std::vector<double> x(n), expected(n), y(n);
  for (std::size_t i = 0; i < n; ++i) x[i] = std::cos(0.9 * static_cast<double>(i));
  // Â x = S A S x.
  std::vector<double> sx(n);
  for (std::size_t i = 0; i < n; ++i) sx[i] = s.scale[i] * x[i];
  spmv_reference(a, sx.data(), expected.data());
  for (std::size_t i = 0; i < n; ++i) expected[i] *= s.scale[i];
  for (const bool sell : {false, true}) {
    const std::unique_ptr<Backend> backend = sell ? make_sell_backend_unit_diagonal(s.off_diagonal)
                                                  : make_seq_backend_unit_diagonal(clone(s.off_diagonal));
    const double dot = backend->spmv_dot(x.data(), y.data());
    double expected_dot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      CHECK_NEAR(y[i], expected[i], 1e-13);
      expected_dot += x[i] * expected[i];
    }
    CHECK_NEAR(dot, expected_dot, 1e-11);
  }
}

TEST(scaled_jacobi_takes_the_iterations_of_jacobi_pcg) {
  for (const Stencil stencil : {Stencil::Poisson2D, Stencil::Poisson3D, Stencil::Point27}) {
    const CsrMatrix a = uneven(stencil, stencil == Stencil::Poisson2D ? 30 : 9);
    const std::size_t n = static_cast<std::size_t>(a.n_rows);
    const AlignedArray<double> b = rhs_for_random_solution(a);
    SolveOptions options;
    options.tol = 1e-10;

    std::vector<double> x_plain(n), x_scaled(n);
    const std::unique_ptr<Backend> plain = make_seq_backend(clone(a));
    const std::unique_ptr<Preconditioner> jacobi = make_jacobi_preconditioner(a);
    const SolveResult expected = verified_cg(a, *plain, *jacobi, b.data(), x_plain.data(), options);

    const JacobiScaling s = jacobi_scaling(a);
    for (const bool sell : {false, true}) {
      const std::unique_ptr<Backend> backend = sell ? make_sell_backend_unit_diagonal(s.off_diagonal)
                                                    : make_seq_backend_unit_diagonal(clone(s.off_diagonal));
      SolveResult result = cg_scaled_jacobi(*backend, s, b.data(), x_scaled.data(), options);
      verify_result(a, b.data(), x_scaled.data(), options, result);
      CHECK_EQ(std::string(to_string(result.status)), std::string("Converged"));
      CHECK(std::llabs(result.spmv_count - expected.spmv_count) <= 1);
      CHECK_NEAR(result.recursive_residual, expected.recursive_residual, 0.05 * expected.recursive_residual);
      for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(x_scaled[i], x_plain[i], 1e-8);
    }
  }
}

TEST(scaled_jacobi_stops_on_the_unscaled_residual) {
  const CsrMatrix a = uneven(Stencil::Poisson2D, 20);
  const AlignedArray<double> b = rhs_for_random_solution(a);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  std::vector<double> x(n);
  const JacobiScaling s = jacobi_scaling(a);
  const std::unique_ptr<Backend> backend = make_seq_backend_unit_diagonal(clone(s.off_diagonal));
  SolveOptions options;
  options.tol = 1e-6;
  SolveResult result = cg_scaled_jacobi(*backend, s, b.data(), x.data(), options);
  verify_result(a, b.data(), x.data(), options, result);
  CHECK(result.recursive_residual < 1e-6);
  CHECK_NEAR(result.true_residual, result.recursive_residual, 1e-9);
}

TEST(scaling_refuses_a_nonpositive_diagonal) {
  Coo coo;
  coo.n_rows = coo.n_cols = 2;
  coo.add(0, 0, 1.0);
  coo.add(1, 1, -1.0);
  CHECK_THROWS_WITH(jacobi_scaling(csr_from_coo(coo)), "positive");
}

TEST(unit_diagonal_backends_refuse_a_stored_diagonal) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 4);
  CHECK_THROWS_WITH(make_seq_backend_unit_diagonal(clone(a)), "without its diagonal");
  CHECK_THROWS_WITH(make_sell_backend_unit_diagonal(a), "without its diagonal");
}

TEST_MAIN()
