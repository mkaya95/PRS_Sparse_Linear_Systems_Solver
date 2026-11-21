// Method 8: Eigen's conjugate gradient with its diagonal (Jacobi) preconditioner, the baseline for method 1.
#include "eigen_cg.hpp"

int main(int argc, char** argv) {
  return ssolve_methods::method_main(argc, argv, "eigen_e1", "Eigen CG + Jacobi",
                                     ssolve_methods::eigen_cg<Eigen::DiagonalPreconditioner<double>>);
}
