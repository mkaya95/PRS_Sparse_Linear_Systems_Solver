// Cache-line aligned array whose elements start uninitialized.
//
// Unlike std::vector, nothing touches the memory at construction. That lets
// OpenMP threads first-touch their own pages later, and keeps vectors from
// being written twice.
#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace ssolve {

template <class T>
class AlignedArray {
  static_assert(std::is_trivially_copyable<T>::value && std::is_trivially_destructible<T>::value,
                "AlignedArray holds plain numeric data only");

 public:
  static constexpr std::size_t kAlignment = 64;

  AlignedArray() noexcept = default;

  explicit AlignedArray(std::size_t n) : size_(n) {
    if (n == 0) return;
    if (n > SIZE_MAX / sizeof(T)) throw std::bad_alloc();
    data_ = static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{kAlignment}));
  }

  ~AlignedArray() { release(); }

  AlignedArray(const AlignedArray&) = delete;
  AlignedArray& operator=(const AlignedArray&) = delete;

  AlignedArray(AlignedArray&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

  AlignedArray& operator=(AlignedArray&& other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }

  T& operator[](std::size_t i) noexcept { return data_[i]; }
  const T& operator[](std::size_t i) const noexcept { return data_[i]; }

  T* begin() noexcept { return data_; }
  T* end() noexcept { return data_ + size_; }
  const T* begin() const noexcept { return data_; }
  const T* end() const noexcept { return data_ + size_; }

  void fill(const T& value) {
    for (std::size_t i = 0; i < size_; ++i) data_[i] = value;
  }

 private:
  void release() noexcept {
    if (data_ != nullptr) ::operator delete(data_, std::align_val_t{kAlignment});
    data_ = nullptr;
    size_ = 0;
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace ssolve
