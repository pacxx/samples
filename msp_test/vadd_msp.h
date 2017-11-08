//
// Created by m_haid02 on 14.08.17.
//

#pragma once

#include <PACXX.h>
using namespace pacxx::v2;

static int test_vadd_msp(int argc, char *argv[]) {

#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
  Executor::Create<NativeRuntime>(0);
#endif

  auto &exec = Executor::get(0);

  size_t size = 128;

  std::vector<int> a(size, 1);
  std::vector<int> b(size, 2);
  std::vector<int> c(size, 0);
  std::vector<int> gold(size, 0);

  auto &da = exec.allocate<int>(a.size(), a.data());
  auto &db = exec.allocate<int>(b.size(), b.data());
  auto &dc = exec.allocate<int>(c.size(), c.data());

  int *pa = da.get();
  int *pb = db.get();
  int *pc = dc.get();

  auto vadd = [=](range &config) {
    auto i = config.get_global(0);
    if (i < _stage([&] { return size; }))
      pc[i] = pa[i] + pb[i] + 2;
  };

  exec.launch(vadd, {{1}, {128}, exec.getID()});
  dc.download(c.data(), c.size());

  std::transform(a.begin(), a.end(), b.begin(), gold.begin(), [](auto a, auto b) { return a + b + 2; });
  if (std::equal(c.begin(), c.end(), gold.begin()))
    return 0;
  else
    return 1;
}
