//
// Created by m_haid02 on 28.06.17.
//

#pragma once

#include <PACXX.h>
using namespace pacxx::v2;

template<typename CFG, typename T>
void kernel(CFG &handle, const T *__restrict__ in, T *__restrict__ out,
            const unsigned int n) {
  [[pacxx::shared]] int sm[16];
  auto i = handle.get_local(0);
  auto g = handle.get_global(0);
  sm[i] = in[g];

  handle.synchronize();

  out[g] = sm[i];

}

static int test_barrier(int argc, char *argv[]) {

  Executor::Create<HIPRuntime>(0);

  auto &exec = Executor::get(0);

  size_t size = 16;

  std::vector<int> a(size * size);
  std::iota(a.begin(), a.end(), 0);
  std::vector<int> b(size * size);

  auto &da = exec.allocate<int>(a.size());
  auto &db = exec.allocate<int>(b.size());

  da.upload(a.data(), a.size());

  auto pa = da.get();
  auto pb = db.get();

  auto vadd = [=](range &config) {
    kernel(config, pa, pb, size);
  };

  exec.launch(vadd, {{size}, {size}});
  db.download(b.data(), b.size());
  for (auto v : b)
    std::cout << v << " "; 
  std::cout << std::endl;
  if (std::equal(a.begin(), a.end(), b.begin()))
    return 0;

  return 1;
}
