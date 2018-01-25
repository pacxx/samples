//
// Created by m_haid02 on 28.06.17.
//

#pragma once

#include <PACXX.h>
using namespace pacxx::v2;

struct my_float4 {
  float x, y, z, w;
};

static int
test_slp(int argc, char *argv[]) {
  auto &exec = Executor::get(0);

  size_t N = 128;

  std::vector<my_float4> a(N);
  std::vector<my_float4> b(a.size());

  for (auto &p : a) {
    p.x = 1;
    p.y = 1;
    p.z = 1;
    p.w = 1;
  }

  auto &da = exec.allocate<my_float4>(a.size());
  auto &db = exec.allocate<my_float4>(b.size());

  da.upload(a.data(), a.size());

  auto pa = da.get();
  auto pb = db.get();

  auto vadd = [=](range &config) {
    int i = config.get_global(0);
    auto r = pa[i];
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        auto a = pa[j];

        r.x += a.x * 2;
        r.y += a.y * 2;
        r.z += a.z * 2;
        r.w += a.w * 2;
      }
    }

    pb[i] = r;
  };

  exec.launch(vadd, {{1}, {a.size()}});
  db.download(b.data(), b.size());

  for (auto p : b)
    std::cout << p.x << " " << p.y << " " << p.z << " " << p.w << std::endl;

  return 0;
}
