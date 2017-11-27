//
// Created by m_haid02 on 28.06.17.
//

#pragma once

#include <PACXX.h>
using namespace pacxx::v2;

int foo(int p, int i) {
  int arr[] = {1, 3, 5, 7, 9, 11, 15, 17};

  if (p == arr[i])
    return i;
  else
    return foo(p, i + 1);
}

static int test_recurse(int argc, char *argv[]) {
  auto &exec = Executor::get(0);

  std::vector<int> a = {1, 3, 5, 7, 9, 11, 15, 17};
  std::vector<int> b(a.size(), 0);
  std::vector<int> gold(a.size(), 0);

  std::iota(gold.begin(), gold.end(), 0);

  DeviceBuffer<int> &da = exec.allocate<int>(a.size());
  da.upload(a.data(), a.size());

  auto pa = da.get();

  auto vadd = [=](range &config) {
    int i = config.get_global(0);
    pa[i] = foo(pa[i], 0);
  };

  exec.launch(vadd, {{1}, {a.size()}});
  da.download(b.data(), b.size());

  if (std::equal(b.begin(), b.end(), gold.begin()))
    return 0;
  else
    return 1;
}
