//
// Created by m_haid02 on 28.06.17.
//

#pragma once

#include <PACXX.h>
#include <pacxx/detail/device/DeviceCode.h>
using namespace pacxx::v2;

static int test_vadd_low_level(int argc, char *argv[]) {
  auto &exec = Executor::get(0);

  size_t size = 128;

  std::vector<int> a(size, 1);
  std::vector<int> b(size, 2);
  std::vector<int> c(size, 0);
  std::vector<int> gold(size, 0);

  DeviceBuffer<int> &da = exec.allocate<int>(a.size());
  auto &db = exec.allocate<int>(b.size());
  auto &dc = exec.allocate<int>(c.size());

  da.upload(a.data(), a.size());
  db.upload(b.data(), b.size());
  dc.upload(c.data(), c.size());

  auto pa = da.get();
  auto pb = db.get();
  auto pc = dc.get();

  auto vadd = [=](range &config) {
    auto i = config.get_global(0);
    if (i < size)
      pc[i] = pa[i] + pb[i] + 2;
  };

  exec.launch(vadd, {{1}, {128}});
  dc.download(c.data(), c.size());

  std::transform(a.begin(), a.end(), b.begin(), gold.begin(), [](auto a, auto b) { return a + b + 2; });
  if (std::equal(c.begin(), c.end(), gold.begin()))
    return 0;
  else
    return 1;
}
