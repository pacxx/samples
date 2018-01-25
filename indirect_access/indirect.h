//
// Created by m_haid02 on 28.06.17.
//

#pragma once

#include <PACXX.h>
#include <algorithm>
using namespace pacxx::v2;

template<typename CFG, typename T, typename IDX>
void kernel(CFG &handle, const T * in, const T* in2,  T* out, IDX idx_map) {
  auto g = handle.get_global(0);

  double grad[2][4] = {{0.0}};  // coordinate x #basisfct
  for (int i = 0; i < 2; i++) // rows of S
    for (int k = 0; k < 2; k++) // columns of S
      for (int j = 0; j < 4; j++) // columns of gradhat
        grad[i][j] += in[k] * in2[k];

  int tmp[4] = {0};
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 4; ++j)
        tmp[k] +=  grad[k][j] * in[idx_map[g][j]];

  out[g] = tmp[g];

}

static int test_indirect(int argc, char *argv[]) {
  auto &exec = Executor::get(0);

  size_t size = 16;

  std::vector<int> a(size * size);
  std::fill(a.begin(), a.end(), 1);
  std::iota(a.begin(), a.end(), 0);
  std::vector<int> b(size * size);
  std::fill(b.begin(), b.end(), 0);
  std::vector<int> c(size * size);
  std::fill(c.begin(), c.end(), 1);
  std::iota(c.begin(), c.end(), 0);

  int idx[256][256];

  for (int j = 0; j < size*size; ++j)
  for (int i = size*size; i > 0; --i)
    idx[j][size*size - i] = i-1;

  auto &da = exec.allocate<int>(a.size());
  auto &db = exec.allocate<int>(b.size());
  auto &dc = exec.allocate<int>(c.size());
  auto &di = exec.allocate<decltype(idx)>(1);

  da.upload(a.data(), a.size());
  db.upload(b.data(), b.size());
  dc.upload(c.data(), c.size());
  di.upload(&idx, 1);

  auto pa = da.get();
  auto pb = db.get();
  auto pc = dc.get();
  auto pi = di.get();

  auto vadd = [=](range &config) {
    kernel(config, pa, pc, pb, *pi);
  };

  exec.launch(vadd, {{1}, {4}});
  db.download(b.data(), b.size());

  for (auto v : b)
    std::cout << v << " ";
  std::cout << std::endl;

  if (std::equal(a.begin(), a.end(), b.begin()))
    return 0;

  return 1;
}
