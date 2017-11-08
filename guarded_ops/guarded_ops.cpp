//
// Created by m_haid02 on 28.04.17.
//

#include <PACXX.h>

#include <iostream>
#include <vector>

using namespace pacxx::v2;

int main() {

#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
  Executor::Create<NativeRuntime>(0);
#endif

  auto &exec = Executor::get(0);

  int i = 0;

  std::vector<int> va(16, 0);
  std::generate(va.begin(), va.end(), [&] { return i++; });
  std::vector<int> vb(16, 1);
  std::vector<int> vc(16, 0);
  auto &a = exec.allocate<int>(160);
  auto &b = exec.allocate<int>(160);
  auto &c = exec.allocate<int>(160);
  a.upload(va.data(), va.size());
  b.upload(vb.data(), vb.size());
  c.upload(vc.data(), vc.size());

  auto pa = a.get();
  auto pb = b.get();
  auto pc = c.get();

  auto lambda = [=](auto &config) {
    auto i = config.get_global(0);
    if (i < 16)
      pc[i] = pa[i] + pb[i];
  };
  Executor::get(0).launch(lambda, {{1}, {16}, 0});

  c.download(vc.data(), 16);

  for (auto v : vc)
    std::cout << v << std::endl;
}
