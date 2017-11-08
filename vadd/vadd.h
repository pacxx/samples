//
// Created by m_haid02 on 28.06.17.
//

#pragma once

#include <gstorm.h>
#include <iostream>
#include <range/v3/all.hpp>
#include <vector>

using namespace ranges;
using namespace gstorm;
using namespace pacxx::v2;

static int test_vadd(int argc, char *argv[]) {

  size_t vsize = 16777216 + 1;

  if (argc >= 2)
    vsize = std::stol(argv[1]);

#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
  Executor::Create<NativeRuntime>(0);
#endif

  std::vector<int> x = view::repeat(1) | view::take(vsize);
  std::vector<int> y = view::repeat(2) | view::take(vsize);
  std::vector<int> gold(vsize);

  auto plus = [](auto &&tpl) { return std::get<0>(tpl) + std::get<1>(tpl); };

  std::vector<int> z;
  auto future = gpu::async(
      [=](const auto &x, const auto &y) {
        return view::zip(x, y) | view::transform(plus);
      },
      x, y);
  z = future.get();

  std::transform(x.begin(), x.end(), y.begin(), gold.begin(), std::plus<int>());
  if (std::equal(z.begin(), z.end(), gold.begin()))
    return 0;
  else
    for (auto v : z)
      std::cout << v <<" ";
  std::cout << std::endl;
  return 1;
}
