//
// Created by m_haid02 on 06.06.17.
//

#include <PACXX.h>
#include <pacxx/detail/cuda/CUDAErrorDetection.h>
#include <cuda_runtime_api.h>

namespace pacxx {
namespace v2 {

template<typename T>
class reference_wrapper {
public:
  // types
  typedef T type;

  // construct/copy/destroy
  reference_wrapper(T &ref) noexcept : _ptr(std::addressof(ref)) {}
  reference_wrapper(T &&) = delete;
  reference_wrapper(const reference_wrapper &) noexcept = default;

  // assignment
  reference_wrapper &operator=(const reference_wrapper &x) noexcept = default;

  // access
  operator T &() const noexcept { return *_ptr; }
  T &get() const noexcept { return *_ptr; }

  auto begin() { return _ptr->begin(); }
  auto end() { return _ptr->end(); }
  auto resize(size_t size) { _ptr->resize(size); }
  size_t size() { return _ptr->size(); }
  auto &operator[](size_t idx) { return _ptr->operator[](idx); }

private:
  T *_ptr;
};
template<typename T, typename... Ts> auto make_managed_ref(Ts &&... args) {
  auto &pvc = Executor::get(0).template allocate<T>(1, MemAllocMode::Unified);
  auto *ptr = pvc.get();
  new(ptr) T(std::forward<Ts>(args)...);
  return reference_wrapper<T>(*ptr);
}

template<typename T> struct ManagedAllocator {
  using value_type = T;

  ManagedAllocator() = default;

  template<class U> ManagedAllocator(const ManagedAllocator<U> &) {}

  T *allocate(std::size_t n) {
    std::cout << "allocate " << n << " bytes\n";
    if (n <= std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      T *ptr = nullptr;
      auto &dp = Executor::get(0).template allocate<T>(n, MemAllocMode::Unified);
      ptr = dp.get();
      if (ptr) {
        return ptr;
      }
    }
    throw std::bad_alloc();
  }

  void deallocate(T *ptr, std::size_t n) { SEC_CUDA_CALL(cudaFree(ptr)); }
};

template<typename T, typename U>
inline bool operator==(const ManagedAllocator<T> &,
                       const ManagedAllocator<U> &) {
  return true;
}

template<typename T, typename U>
inline bool operator!=(const ManagedAllocator<T> &a,
                       const ManagedAllocator<U> &b) {
  return !(a == b);
}
}
}

using namespace pacxx::v2;

int main() {

  int x = 5;
  auto a = make_managed_ref<std::vector<int, ManagedAllocator<int>>>(16);
  auto b = make_managed_ref<std::vector<int, ManagedAllocator<int>>>(16);
  auto c = make_managed_ref<std::vector<int, ManagedAllocator<int>>>(16);

  c.resize(16);
  std::fill(a.begin(), a.end(), 1);
  std::fill(b.begin(), b.end(), 2);
  std::fill(c.begin(), c.end(), -1);

  auto saxpy = [=](auto &handle) mutable {
    auto idx = handle.get_global(0);
    if (idx >= c.size())
      return;
    auto out = c.begin();

    auto ina = a.begin();
    auto inb = b.begin();
    std::advance(out, idx);
    std::advance(ina, idx);
    std::advance(inb, idx);
    *out = x * *ina + *inb + 3;
  };

  Executor::get(0).launch(saxpy, {{1}, {c.size()}, 0});

  SEC_CUDA_CALL(cudaDeviceSynchronize());

  for (auto v : c)
    std::cout << v << " ";
  std::cout << std::endl;
}
