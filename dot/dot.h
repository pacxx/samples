//
// Created by m_haid02 on 28.06.17.
//

#pragma once

#include <gstorm.h>
#include <range/v3/all.hpp>

#define COMPARE_SEQUENTIAL

using namespace ranges::v3;
using namespace gstorm;
using namespace pacxx::v2;

static int test_dot(int argc, char *argv[]) {
  size_t vsize = 4096 * 4096;
  int runs = 0;

  if (argc >= 2)
    vsize = std::stol(argv[1]);

  if (argc >= 3)
    runs = std::stoi(argv[2]);

#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
  Executor::Create<NativeRuntime>(0);
#endif

  std::vector<int> xs = view::iota(0) | view::take(vsize);
  std::vector<int> ys = view::repeat(1) | view::take(vsize);

  auto gxs = xs | gpu::copy;
  auto gys = ys | gpu::copy;
  auto xys = view::zip(gxs, gys) | view::transform([](auto &&tpl) {
    return std::get<0>(tpl) * std::get<1>(tpl);
  });

  int sum = 0;
  sum = gpu::algorithm::reduce(xys, 0, [](auto &&lhs, auto &&rhs) { return lhs + rhs; });

#ifdef COMPARE_SEQUENTIAL
  int gold = std::inner_product(xs.begin(), xs.end(), ys.begin(), 0,
                                std::plus<>(), std::multiplies<>());
  if (gold == sum) {
    std::cout << "success" << std::endl;
    return 0;
  } else {
    std::cout << "failed! got " << sum << " expected " << gold << std::endl;
    return 1;
  }
#endif
}
