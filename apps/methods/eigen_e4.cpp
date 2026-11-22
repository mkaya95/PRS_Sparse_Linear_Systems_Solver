// Method 9: Eigen's conjugate gradient with its incomplete Cholesky in natural order, the baseline for method 4.
#include "eigen_cg.hpp"

int main(int argc, char** argv) {
  return ssolve_methods::method_main(
      argc, argv, "eigen_e4", "Eigen CG + incomplete Cholesky, natural order",
      ssolve_methods::eigen_cg<Eigen::IncompleteCholesky<double, Eigen::Lower, Eigen::NaturalOrdering<int>>>);
}
