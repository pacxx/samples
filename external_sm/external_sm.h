//
// Created by m_haid02 on 28.06.17.
//

#pragma once

#include <PACXX.h>
using namespace pacxx::v2;

static int test_external_sm(int argc, char *argv[]) {
  auto &exec = Executor::get(0);

  size_t size = 256;

  std::vector<int> a(size * size);
  std::iota(a.begin(), a.end(), 0);
  std::vector<int> b(size * size);

  auto &da = exec.allocate<int>(a.size());
  auto &db = exec.allocate<int>(b.size());

  da.upload(a.data(), a.size());

  auto pa = da.get();
  auto pb = db.get();

  auto vadd = [=](range &handle) {
  [[pacxx::shared]] extern int sm[];
  auto i = handle.get_local(0);
  auto g = handle.get_global(0);
  sm[i] = pa[g];

  handle.synchronize();

  pb[g] = sm[i];
  };

  exec.launch(vadd, {{size}, {size}, size * sizeof(int)});
  db.download(b.data(), b.size());
  if (std::equal(a.begin(), a.end(), b.begin()))
    return 0;

  return 1;
}
