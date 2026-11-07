// Convergence history for plots. It wraps the backend rather than changing cg(), so the loop
// the benchmarks time stays exactly as it was measured.
#include <cmath>
#include <stdexcept>
#include <vector>

#include "ssolve/cg.hpp"

namespace ssolve {

namespace {

// Forwards every call to the real backend and keeps the r·r that each residual update returns.
class RecordingBackend final : public Backend {
 public:
  RecordingBackend(const Backend& inner, std::vector<double>& rr) : inner_(inner), rr_(rr) {}

  Index n() const override { return inner_.n(); }
  void set_zero(double* x) const override { inner_.set_zero(x); }
  void copy(const double* source, double* destination) const override { inner_.copy(source, destination); }
  double dot(const double* x, const double* y) const override { return inner_.dot(x, y); }
  void spmv(const double* x, double* y) const override { inner_.spmv(x, y); }
  void xpby(const double* x, double beta, double* y) const override { inner_.xpby(x, beta, y); }
  double spmv_dot(const double* x, double* y) const override { return inner_.spmv_dot(x, y); }

  double cg_update(double a, const double* p, const double* q, double* x, double* r) const override {
    const double rr = inner_.cg_update(a, p, q, x, r);
    rr_.push_back(rr);
    return rr;
  }

  RrRz jacobi_update(double a, const double* p, const double* q, double* x, double* r,
                     const double* inv_diag) const override {
    const RrRz sums = inner_.jacobi_update(a, p, q, x, r, inv_diag);
    rr_.push_back(sums.rr);
    return sums;
  }

  void jacobi_direction(const double* inv_diag, const double* r, double beta, double* p) const override {
    inner_.jacobi_direction(inv_diag, r, beta, p);
  }

 private:
  const Backend& inner_;
  std::vector<double>& rr_;
};

}  // namespace

SolveResult cg_with_history(const Backend& backend, const double* inv_diag, const double* b, double* x,
                            const SolveOptions& options, std::vector<double>& history) {
  if (options.use_initial_guess) throw std::runtime_error("cg_with_history starts from x = 0");
  std::vector<double> rr;
  const RecordingBackend recording(backend, rr);
  const SolveResult result = cg(recording, inv_diag, b, x, options);

  history.clear();
  if (rr.empty()) {
    // No iteration ran. b = 0 reports 0 and gets no history; a solve that met tol at once starts at 1.
    if (result.status == SolveStatus::Converged && result.recursive_residual > 0.0) history.push_back(1.0);
    return result;
  }
  // From x = 0 the first residual is b itself. cg() may have solved with a rescaled b, so ‖b‖² in
  // that same scale comes from its final report: recursive_residual = sqrt(last r·r / ‖b‖²).
  const double last = rr.back();
  const double reported = result.recursive_residual;
  const double bb = (std::isfinite(reported) && reported > 0.0 && std::isfinite(last)) ? last / (reported * reported)
                                                                                         : recording.dot(b, b);
  history.push_back(1.0);
  for (const double value : rr) history.push_back(std::sqrt(value / bb));
  return result;
}

}  // namespace ssolve
