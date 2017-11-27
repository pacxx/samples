//
// Created by m_haid02 on 28.06.17.
//

#pragma once

#include <PACXX.h>
using namespace pacxx::v2;

static int test_printf(int argc, char *argv[]) {
  auto &exec = Executor::get(0);

  auto kernel = [=](range &config) {
    pacxx::printf("hello world %i %i\n", config.get_global(0), config.get_block(0));
  };

  exec.launch(kernel, {{1}, {1}});
  exec.synchronize();
  return 0;
}
