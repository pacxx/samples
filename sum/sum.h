//
// Created by m_haid02 on 28.06.17.
//

#pragma once

#include <gstorm.h>
#include <range/v3/all.hpp>

using namespace ranges::v3;
using namespace gstorm;
using namespace pacxx::v2;

int rng = 0;

static int test_sum(int argc, char *argv[]) {
  size_t vsize = 256;// 4096 * 4096;
  int runs = 10;

  if (argc >= 2)
    vsize = std::stol(argv[1]);

  if (argc >= 3)
    runs = std::stoi(argv[2]);

#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
  Executor::Create<NativeRuntime>(0);
#endif

  std::vector<int> h_vec(vsize);
  std::fill(h_vec.begin(), h_vec.end(), 1);

  auto d_vec = h_vec | gpu::copy;
  int sum = 0;
  sum = gpu::algorithm::reduce(
      d_vec, 0, [](auto &&lhs, auto &&rhs) { return lhs + rhs; });

  std::cout << "sum is " << sum << std::endl;

  int gold = std::accumulate(h_vec.begin(), h_vec.end(), 0, std::plus<>());
  if (gold == sum)
    return 0;
  else
    return 1;
}
