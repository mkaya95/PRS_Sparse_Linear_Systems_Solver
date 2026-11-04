#include <cstdint>
#include <new>

#include "ssolve/aligned_array.hpp"
#include "test_util.hpp"

using ssolve::AlignedArray;

TEST(data_is_64_byte_aligned_for_every_size) {
  for (std::size_t n : {1u, 7u, 8u, 9u, 63u, 64u, 65u, 4096u}) {
    AlignedArray<double> a(n);
    CHECK_EQ(a.size(), n);
    CHECK_EQ(reinterpret_cast<std::uintptr_t>(a.data()) % 64u, std::uintptr_t{0});
  }
}

TEST(empty_arrays_iterate_zero_times) {
  AlignedArray<double> by_default;
  AlignedArray<double> zero_length(0);
  CHECK_EQ(by_default.size(), std::size_t{0});
  CHECK_EQ(zero_length.size(), std::size_t{0});
  CHECK(by_default.begin() == by_default.end());
  CHECK(zero_length.begin() == zero_length.end());
}

TEST(move_construct_transfers_the_buffer) {
  AlignedArray<int> source(10);
  int* const original = source.data();
  AlignedArray<int> target(std::move(source));
  CHECK(target.data() == original);
  CHECK_EQ(target.size(), std::size_t{10});
  CHECK(source.data() == nullptr);
  CHECK_EQ(source.size(), std::size_t{0});
}

TEST(move_assign_over_a_non_empty_array_frees_the_old_buffer) {
  AlignedArray<int> target(100);  // ASan reports a leak if this buffer is not freed
  AlignedArray<int> source(5);
  int* const original = source.data();
  target = std::move(source);
  CHECK(target.data() == original);
  CHECK_EQ(target.size(), std::size_t{5});
  CHECK(source.data() == nullptr);
}

TEST(fill_writes_every_element) {
  AlignedArray<double> a(9);
  a.fill(3.5);
  CHECK_EQ(a[0], 3.5);
  CHECK_EQ(a[8], 3.5);
  double sum = 0.0;
  for (double v : a) sum += v;
  CHECK_EQ(sum, 31.5);
}

TEST(const_access_reads_the_same_elements) {
  AlignedArray<double> a(3);
  a.fill(2.0);
  const AlignedArray<double>& view = a;
  CHECK_EQ(view[2], 2.0);
  CHECK(view.data() == a.data());
  CHECK(view.end() - view.begin() == 3);
}

TEST(overflowing_size_throws_instead_of_wrapping) {
  bool threw = false;
  try {
    AlignedArray<double> huge(SIZE_MAX / 2);
  } catch (const std::bad_alloc&) {
    threw = true;
  }
  CHECK(threw);
}

TEST_MAIN()
