#include "ssolve/generators.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

namespace ssolve {

namespace {

struct Offset {
  int di, dj, dl;
};

// Offsets in lexicographic order, which is ascending column order for k = (i·N + j)·Nz + l.
std::vector<Offset> stencil_offsets(Stencil stencil) {
  switch (stencil) {
    case Stencil::Poisson2D:
      return {{-1, 0, 0}, {0, -1, 0}, {0, 0, 0}, {0, 1, 0}, {1, 0, 0}};
    case Stencil::Poisson3D:
      return {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}};
    case Stencil::Point27: {
      std::vector<Offset> all;
      for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
          for (int dl = -1; dl <= 1; ++dl) all.push_back({di, dj, dl});
        }
      }
      return all;
    }
  }
  throw std::logic_error("unknown stencil");
}

double stencil_diagonal(Stencil stencil) {
  switch (stencil) {
    case Stencil::Poisson2D:
      return 4.0;
    case Stencil::Poisson3D:
      return 6.0;
    case Stencil::Point27:
      return 26.0;
  }
  throw std::logic_error("unknown stencil");
}

bool is_2d(Stencil stencil) { return stencil == Stencil::Poisson2D; }

// Largest N whose row count still fits in Index.
Index max_grid_side(Stencil stencil) {
  const std::int64_t limit = std::numeric_limits<Index>::max();
  std::int64_t N = 1;
  while (true) {
    const std::int64_t next = N + 1;
    const std::int64_t rows = is_2d(stencil) ? next * next : next * next * next;
    if (rows > limit) return static_cast<Index>(N);
    N = next;
  }
}

}  // namespace

std::int64_t stencil_nnz(Stencil stencil, std::int64_t N) {
  switch (stencil) {
    case Stencil::Poisson2D:
      return 5 * N * N - 4 * N;
    case Stencil::Poisson3D:
      return 7 * N * N * N - 6 * N * N;
    case Stencil::Point27:
      return (3 * N - 2) * (3 * N - 2) * (3 * N - 2);
  }
  throw std::logic_error("unknown stencil");
}

CsrMatrix make_stencil(Stencil stencil, Index N) {
  if (N <= 0) throw std::runtime_error("grid side N must be positive");
  if (N > max_grid_side(stencil)) throw std::runtime_error("grid is too large for 32-bit indices");

  const std::vector<Offset> offsets = stencil_offsets(stencil);
  const double diagonal = stencil_diagonal(stencil);
  const Index layers = is_2d(stencil) ? 1 : N;  // grid points along the l axis
  const std::int64_t rows = static_cast<std::int64_t>(N) * N * layers;

  const auto inside = [&](Index i, Index j, Index l, const Offset& o) {
    const Index ni = i + o.di, nj = j + o.dj, nl = l + o.dl;
    return ni >= 0 && ni < N && nj >= 0 && nj < N && nl >= 0 && nl < layers;
  };

  CsrMatrix a;
  a.n_rows = static_cast<Index>(rows);
  a.n_cols = static_cast<Index>(rows);
  a.row_ptr = AlignedArray<std::int64_t>(static_cast<std::size_t>(rows) + 1);

  // row_ptr first and rows second, so the second loop can later run in parallel for first-touch.
  a.row_ptr[0] = 0;
  std::int64_t k = 0;
  for (Index i = 0; i < N; ++i) {
    for (Index j = 0; j < N; ++j) {
      for (Index l = 0; l < layers; ++l, ++k) {
        std::int64_t count = 0;
        for (const Offset& o : offsets) count += inside(i, j, l, o) ? 1 : 0;
        a.row_ptr[k + 1] = a.row_ptr[k] + count;
      }
    }
  }

  a.col = AlignedArray<Index>(static_cast<std::size_t>(a.row_ptr[rows]));
  a.val = AlignedArray<double>(static_cast<std::size_t>(a.row_ptr[rows]));
  k = 0;
  for (Index i = 0; i < N; ++i) {
    for (Index j = 0; j < N; ++j) {
      for (Index l = 0; l < layers; ++l, ++k) {
        std::int64_t position = a.row_ptr[k];
        for (const Offset& o : offsets) {
          if (!inside(i, j, l, o)) continue;
          const std::int64_t column =
              (static_cast<std::int64_t>(i + o.di) * N + (j + o.dj)) * layers + (l + o.dl);
          a.col[position] = static_cast<Index>(column);
          a.val[position] = column == k ? diagonal : -1.0;
          ++position;
        }
      }
    }
  }
  return a;
}

AlignedArray<double> rhs_for_unit_solution(const CsrMatrix& a) {
  AlignedArray<double> b(static_cast<std::size_t>(a.n_rows));
  for (Index i = 0; i < a.n_rows; ++i) {
    double sum = 0.0;
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) sum += a.val[k];
    b[i] = sum;
  }
  return b;
}

}  // namespace ssolve
