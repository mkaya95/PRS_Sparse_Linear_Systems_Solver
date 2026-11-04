#pragma once

#include <chrono>

namespace ssolve {

inline double wall_seconds() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

}  // namespace ssolve
