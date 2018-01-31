//
// Created by m_haid02 on 06.06.17.
//

#include <PACXX.h>
#include <memory>

namespace pacxx {
namespace v2 {

template <typename T> class ref {
public:
  // types
  typedef T type;

  // construct/copy/destroy
  ref(T &ref, Executor &exec) noexcept
      : _ptr(std::addressof(ref), [&](auto ptr) {
          ptr->T::~T();
          exec.template free<T>(ptr);
        }) {}

  ref(T &&) = delete;
  ref(const ref &) noexcept = default;

  // assignment
  ref &operator=(const ref &x) noexcept = default;

  // access
  operator T &() const noexcept { return *_ptr; }
  T &get() const noexcept { return *_ptr; }

  auto begin() { return _ptr->begin(); }
  auto end() { return _ptr->end(); }
  auto resize(size_t size) { _ptr->resize(size); }
  size_t size() { return _ptr->size(); }
  auto &operator[](size_t idx) { return _ptr->operator[](idx); }

private:
  std::shared_ptr<T> _ptr;
};

template <typename T> struct managed_allocator {
  using value_type = T;

  managed_allocator(Executor &exec) : _exec(exec) {}

  template <class U> managed_allocator(const managed_allocator<U>& other) {
    _exec = other._exec; 
  }

  T *allocate(std::size_t n) {
    if (n <= std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      auto *ptr = _exec.template allocate<T>(n, MemAllocMode::Unified).get();
      if (ptr) {
        return ptr;
      }
    }
    throw std::bad_alloc();
  }

  void deallocate(T *ptr, std::size_t n) { _exec.template free<T>(ptr); }

private:
  Executor &_exec;
};

template <typename T, typename U>
inline bool operator==(const managed_allocator<T> &,
                       const managed_allocator<U> &) {
  return true;
}

template <typename T, typename U>
inline bool operator!=(const managed_allocator<T> &a,
                       const managed_allocator<U> &b) {
  return !(a == b);
}

template <template <typename, typename> class Container, typename T,
          typename... Ts>
auto make_managed_ref(Executor &exec, Ts &&... args) {
  using ContainerTy = Container<T, managed_allocator<T>>;

  auto *ptr = exec.template allocate<Container<T, managed_allocator<T>>>(
                      1, MemAllocMode::Unified)
                  .get();
  new (ptr) ContainerTy(std::forward<Ts>(args)..., managed_allocator<T>(exec));

  return ref<ContainerTy>(*ptr, exec);
}
} // namespace v2
} // namespace pacxx

using namespace pacxx::v2;

int main() {

  int x = 5;
  auto &exec = Executor::get(0);
  auto a = make_managed_ref<std::vector, int>(exec, 16);
  auto b = make_managed_ref<std::vector, int>(exec, 16);
  auto c = make_managed_ref<std::vector, int>(exec);

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

  exec.launch(saxpy, {{1}, {c.size()}, 0});
  exec.synchronize();

  for (auto v : c)
    std::cout << v << " ";
  std::cout << std::endl;
}
