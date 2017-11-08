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

namespace gstorm {
namespace functional {
auto plus = [](auto &&tpl) { return std::get<0>(tpl) + std::get<1>(tpl); };
auto multiplies = [](auto &&tpl) {
  return std::get<0>(tpl) * std::get<1>(tpl);
};

auto saxpy = [](auto &&tpl) {
  return std::get<0>(tpl) * std::get<1>(tpl) + std::get<2>(tpl);
};
}
}

auto saxpy_fast(float A, std::vector<float> &X, std::vector<float> &Y) {
  return gpu::async(
      [=](const auto &a, const auto &x, const auto &y) {
        return view::zip(view::repeat(a), x, y) |
            view::transform(functional::saxpy);
      },
      A, X, Y);
}
auto saxpy_slow(float A, std::vector<float> &X, std::vector<float> &Y) {
  return gpu::async(
      [=](const auto &a, const auto &x, const auto &y) {
        auto ax =
            view::zip(view::repeat(a), x) | view::transform(functional::multiplies);
        return view::zip(ax, y) | view::transform(functional::plus);
      },
      A, X, Y);
}

static int test_saxpy(int argc, char *argv[]) {

  size_t vsize = 4096 * 4096;

  if (argc >= 2)
    vsize = std::stol(argv[1]);

#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
  Executor::Create<NativeRuntime>(0);
#endif

  std::vector<float> x = view::repeat(1.0f) | view::take(vsize);
  std::vector<float> y = view::repeat(2.0f) | view::take(vsize);
  std::vector<float> gold(vsize);
  saxpy_slow(2.0f, x, y).wait();
  std::vector<float> z = saxpy_fast(2.0f, x, y).get();

  std::transform(x.begin(), x.end(), y.begin(), gold.begin(),
                 [a = 2.0f](auto x, auto y) { return a * x + y; });
  if (std::equal(z.begin(), z.end(), gold.begin()))
    return 0;
  else
    return 1;
}
