// Minimal test framework: register cases with TEST, check with CHECK_*, finish with TEST_MAIN.
#pragma once

#include <cmath>
#include <cstdio>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace test_util {

struct Case {
  const char* name;
  void (*fn)();
};

inline std::vector<Case>& cases() {
  static std::vector<Case> all;
  return all;
}

inline int& failures() {
  static int count = 0;
  return count;
}

struct Register {
  Register(const char* name, void (*fn)()) { cases().push_back({name, fn}); }
};

template <class T>
std::string str(const T& value) {
  std::ostringstream out;
  out.precision(17);
  out << value;
  return out.str();
}

inline void fail(const char* file, int line, const std::string& message) {
  std::printf("  %s:%d: %s\n", file, line, message.c_str());
  ++failures();
}

inline int run_all() {
  int failed_cases = 0;
  for (const Case& c : cases()) {
    const int before = failures();
    try {
      c.fn();
    } catch (const std::exception& e) {
      fail(c.name, 0, std::string("unexpected exception: ") + e.what());
    } catch (...) {
      fail(c.name, 0, "unexpected non-standard exception");
    }
    if (failures() != before) {
      std::printf("FAILED %s\n", c.name);
      ++failed_cases;
    }
  }
  std::printf("%zu tests, %d failed\n", cases().size(), failed_cases);
  return failed_cases == 0 ? 0 : 1;
}

}  // namespace test_util

#define TEST(name)                                                 \
  static void test_##name();                                       \
  static const ::test_util::Register reg_##name(#name, test_##name); \
  static void test_##name()

#define CHECK(cond)                                                \
  do {                                                             \
    if (!(cond)) ::test_util::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
  } while (0)

#define CHECK_EQ(a, b)                                                                   \
  do {                                                                                   \
    const auto check_a_ = (a);                                                           \
    const auto check_b_ = (b);                                                           \
    if (!(check_a_ == check_b_))                                                         \
      ::test_util::fail(__FILE__, __LINE__,                                              \
                        std::string(#a " == " #b ": ") + ::test_util::str(check_a_) +    \
                            " vs " + ::test_util::str(check_b_));                        \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                            \
  do {                                                                                   \
    const double check_a_ = (a);                                                         \
    const double check_b_ = (b);                                                         \
    if (!(std::fabs(check_a_ - check_b_) <= (tol)))                                      \
      ::test_util::fail(__FILE__, __LINE__,                                              \
                        std::string(#a " ~= " #b ": ") + ::test_util::str(check_a_) +    \
                            " vs " + ::test_util::str(check_b_));                        \
  } while (0)

#define CHECK_THROWS_WITH(expr, substring)                                               \
  do {                                                                                   \
    bool check_threw_ = false;                                                           \
    try {                                                                                \
      (void)(expr);                                                                      \
    } catch (const std::exception& check_e_) {                                           \
      check_threw_ = true;                                                               \
      if (std::string(check_e_.what()).find(substring) == std::string::npos)             \
        ::test_util::fail(__FILE__, __LINE__,                                            \
                          std::string("message lacks '") + (substring) + "': " + check_e_.what()); \
    }                                                                                    \
    if (!check_threw_)                                                                   \
      ::test_util::fail(__FILE__, __LINE__, "expected a throw from " #expr);             \
  } while (0)

#define TEST_MAIN() \
  int main() { return ::test_util::run_all(); }
