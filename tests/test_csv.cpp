#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "ssolve/csv.hpp"
#include "ssolve/manifest.hpp"
#include "ssolve/timer.hpp"
#include "test_util.hpp"

using namespace ssolve;

namespace {

std::string read_file(const std::string& path) {
  std::ifstream file(path);
  std::stringstream text;
  text << file.rdbuf();
  return text.str();
}

}  // namespace

TEST(plain_text_is_not_escaped) { CHECK_EQ(csv_escape("abc 1.5"), std::string("abc 1.5")); }

TEST(comma_forces_quotes) { CHECK_EQ(csv_escape("a,b"), std::string("\"a,b\"")); }

TEST(embedded_quote_is_doubled) { CHECK_EQ(csv_escape("a\"b"), std::string("\"a\"\"b\"")); }

TEST(newline_forces_quotes) { CHECK_EQ(csv_escape("a\nb"), std::string("\"a\nb\"")); }

TEST(two_writers_share_one_header) {
  const std::string path = "test_csv_output.csv";
  std::remove(path.c_str());
  {
    CsvWriter first(path, {"a", "b"});
    first.write_row({"1", "2"});
  }
  {
    CsvWriter second(path, {"a", "b"});
    second.write_row({"3", "x,y"});
  }
  CHECK_EQ(read_file(path), std::string("a,b\n1,2\n3,\"x,y\"\n"));
  std::remove(path.c_str());
}

TEST(wrong_value_count_throws) {
  const std::string path = "test_csv_count.csv";
  std::remove(path.c_str());
  CsvWriter writer(path, {"a", "b"});
  CHECK_THROWS_WITH(writer.write_row({"1"}), "2 columns");
  std::remove(path.c_str());
}

TEST(unwritable_path_throws_naming_the_path) {
  CHECK_THROWS_WITH(CsvWriter("no/such/dir/x.csv", {"a"}), "no/such/dir/x.csv");
}

TEST(fmt_double_round_trips_exactly) {
  for (double value : {0.1, 1.0 / 3.0, 1e-300, 12345.678, 0.0}) {
    CHECK_EQ(std::stod(fmt_double(value)), value);
  }
}

TEST(manifest_describes_this_machine) {
  const Manifest m = collect_manifest();
  CHECK(m.logical_cpus >= 1);
  CHECK(m.physical_cores >= 0);
  CHECK(m.l3_bytes >= 0);
  CHECK(m.mem_bytes > 0);
  CHECK(!m.compiler.empty());
  CHECK(m.flags.find("-fopenmp-simd") != std::string::npos);
  CHECK(!m.git_commit.empty());
}

TEST(manifest_columns_and_values_line_up) {
  CHECK_EQ(manifest_columns().size(), manifest_values(collect_manifest()).size());
}

TEST(timer_moves_forward) {
  const double start = wall_seconds();
  CHECK(wall_seconds() >= start);
}

TEST_MAIN()
