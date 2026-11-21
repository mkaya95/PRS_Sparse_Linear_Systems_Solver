// Convergence history for plots. It wraps the preconditioner rather than changing cg(), so the loop
// the benchmarks time stays exactly as it was measured.
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ssolve/cg.hpp"

namespace ssolve {

namespace {

// Forwards every call to the real preconditioner and keeps the r·r that each residual update returns.
// It wraps the preconditioner rather than the backend, because a preconditioner may update r in its own
// fused loop (triangular.hpp) instead of the backend's.
class RecordingPreconditioner final : public Preconditioner {
 public:
  RecordingPreconditioner(Preconditioner& inner, std::vector<double>& rr) : inner_(inner), rr_(rr) {}

  double start(const Backend& backend, const double* r, double rr, double* p) override {
    return inner_.start(backend, r, rr, p);
  }

  RrRz update(const Backend& backend, double a, const double* p, const double* q, double* x, double* r) override {
    const RrRz sums = inner_.update(backend, a, p, q, x, r);
    rr_.push_back(sums.rr);
    return sums;
  }

  void direction(const Backend& backend, const double* r, double beta, double* p) override {
    inner_.direction(backend, r, beta, p);
  }

  double norm_squared(const Backend& backend, const double* v) override { return inner_.norm_squared(backend, v); }

 private:
  Preconditioner& inner_;
  std::vector<double>& rr_;
};

}  // namespace

SolveResult cg_with_history(const Backend& backend, const double* inv_diag, const double* b, double* x,
                            const SolveOptions& options, std::vector<double>& history) {
  if (options.use_initial_guess) throw std::runtime_error("cg_with_history starts from x = 0");
  const std::unique_ptr<Preconditioner> precond = preconditioner_for(backend, inv_diag);
  return cg_with_history(backend, *precond, b, x, options, history);
}

SolveResult cg_with_history(const Backend& backend, Preconditioner& precond, const double* b, double* x,
                            const SolveOptions& options, std::vector<double>& history) {
  if (options.use_initial_guess) throw std::runtime_error("cg_with_history starts from x = 0");
  std::vector<double> rr;
  RecordingPreconditioner recording(precond, rr);
  SolveOptions plain = options;
  plain.history = nullptr;  // this function is the recorder; cg() must not call back into it
  const SolveResult result = cg(backend, recording, b, x, plain);

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
                                                                                         : precond.norm_squared(backend, b);
  history.push_back(1.0);
  for (const double value : rr) history.push_back(std::sqrt(value / bb));
  return result;
}

}  // namespace ssolve
