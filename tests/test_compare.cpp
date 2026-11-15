#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "ssolve/compare.hpp"
#include "test_util.hpp"

using namespace ssolve;

namespace {

std::vector<double> ramp(std::size_t n) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = 0.5 + 0.001 * static_cast<double>(i);
  return v;
}

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

}  // namespace

TEST(identical_vectors_differ_nowhere) {
  const std::vector<double> a = ramp(5000);
  const VectorDifference d = compare_vectors("ours", a.data(), "eigen", a.data(), a.size());
  CHECK_EQ(d.a, std::string("ours"));
  CHECK_EQ(d.b, std::string("eigen"));
  CHECK_EQ(d.max_abs, 0.0);
  CHECK_EQ(d.rel_norm, 0.0);
  CHECK_EQ(d.equal, std::int64_t{5000});
  CHECK_EQ(d.cells.size(), std::size_t{64 * 64});
  for (const double cell : d.cells) CHECK_EQ(cell, 0.0);
}

TEST(one_changed_entry_shows_in_its_cell_only) {
  const std::vector<double> b = ramp(5000);
  std::vector<double> a = b;
  a[1234] = b[1234] + 1e-9;
  const VectorDifference d = compare_vectors("a", a.data(), "b", b.data(), a.size());
  const std::size_t changed = 1234 * 4096 / 5000;
  for (std::size_t c = 0; c < d.cells.size(); ++c) {
    if (c == changed) {
      CHECK_EQ(d.cells[c], a[1234] - b[1234]);
    } else {
      CHECK_EQ(d.cells[c], 0.0);
    }
  }
  CHECK_EQ(d.max_abs, a[1234] - b[1234]);
  CHECK_NEAR(d.max_abs, 1e-9, 1e-15);
  CHECK_EQ(d.equal, std::int64_t{4999});
}

TEST(a_small_vector_gets_one_cell_per_entry) {
  CHECK_EQ(difference_grid(5), 3);
  CHECK_EQ(difference_grid(1), 1);
  CHECK_EQ(difference_grid(4096), 64);
  CHECK_EQ(difference_grid(1000000), 64);
  const std::vector<double> a = {1, 2, 3, 4, 5};
  const std::vector<double> b = {1, 2.5, 3, 3, 5};
  const VectorDifference d = compare_vectors("a", a.data(), "b", b.data(), a.size());
  CHECK_EQ(d.cells.size(), std::size_t{9});
  CHECK_EQ(d.cells[0], 0.0);
  CHECK_EQ(d.cells[1], 0.5);
  CHECK_EQ(d.cells[2], 0.0);
  CHECK_EQ(d.cells[3], 1.0);
  CHECK_EQ(d.cells[4], 0.0);
  for (std::size_t c = 5; c < 9; ++c) CHECK(std::isnan(d.cells[c]));
}

TEST(a_non_finite_difference_is_marked) {
  const std::vector<double> a = {1, 2, 3, 4};
  std::vector<double> b = a;
  b[2] = std::numeric_limits<double>::quiet_NaN();
  const VectorDifference d = compare_vectors("a", a.data(), "b", b.data(), a.size());
  CHECK_EQ(d.cells[2], -1.0);
  CHECK_EQ(d.cells[1], 0.0);
  CHECK(std::isnan(d.max_abs));
  CHECK(std::isnan(d.rel_norm));
}

TEST(rel_norm_uses_the_second_vector) {
  const std::vector<double> a = {3, 4}, zero = {0, 0};
  CHECK_EQ(compare_vectors("a", a.data(), "b", zero.data(), 2).rel_norm, 5.0);
  const std::vector<double> ones = {1, 1}, twos = {2, 2};
  CHECK_EQ(compare_vectors("a", ones.data(), "b", twos.data(), 2).rel_norm, 0.5);
  // Huge entries must not overflow while squaring.
  const std::vector<double> big = {1e300, 1e300}, bigger = {2e300, 2e300};
  CHECK_NEAR(compare_vectors("a", big.data(), "b", bigger.data(), 2).rel_norm, 0.5, 1e-15);
}

TEST(json_has_every_field_and_null_for_empty_cells) {
  const std::vector<double> a = {1, 2, 3, 4, 5};
  const std::vector<double> b = {1, 2, 3, 4, 6};
  const VectorDifference d = compare_vectors("x\"y", a.data(), "true", b.data(), a.size());
  const std::string json = solutions_json("m", 5, "random", 1.0, {d});
  CHECK(contains(json, "\"matrix\":\"m\""));
  CHECK(contains(json, "\"n\":5"));
  CHECK(contains(json, "\"grid\":3"));
  CHECK(contains(json, "\"problem\":\"random\""));
  CHECK(contains(json, "\"x_scale\":1"));
  CHECK(contains(json, "\"comparisons\":[{"));
  CHECK(contains(json, "\"a\":\"x\\\"y\""));
  CHECK(contains(json, "\"b\":\"true\""));
  CHECK(contains(json, "\"equal\":4"));
  CHECK(contains(json, "\"cells\":[0,0,0,0,1,null,null,null,null]"));
}

TEST(json_lists_each_convergence_history) {
  const std::string none = solutions_json("m", 2, "unit", 2.0, {});
  CHECK(contains(none, "\"comparisons\":[],\"histories\":[]}"));
  const std::string json = solutions_json("m", 2, "unit", 2.0, {}, {{"ours", {1.0, 0.123456, 1e-9}}, {"b", {}}});
  CHECK(contains(json, "\"histories\":[{\"solver\":\"ours\",\"residuals\":[1,0.1235,1e-09]},"
                       "{\"solver\":\"b\",\"residuals\":[]}]}"));
}

TEST(json_writes_null_for_a_non_finite_summary) {
  const std::vector<double> a = {1, 2};
  const std::vector<double> b = {1, std::numeric_limits<double>::infinity()};
  const VectorDifference d = compare_vectors("a", a.data(), "b", b.data(), a.size());
  const std::string json = solutions_json("m", 2, "unit", 2.0, {d});
  CHECK(contains(json, "\"max_abs\":null"));
  CHECK(contains(json, "\"rel_norm\":null"));
  CHECK(contains(json, "\"cells\":[0,-1,null,null]"));
}

TEST(atomic_write_leaves_no_partial_file) {
  const std::string path = "/tmp/ssolve-test-compare-" + std::to_string(getpid()) + ".json";
  write_file_atomically(path, "{\"ok\":true}\n");
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  CHECK_EQ(text.str(), std::string("{\"ok\":true}\n"));
  CHECK(!std::ifstream(path + ".part").good());
  std::remove(path.c_str());
}

TEST_MAIN()
