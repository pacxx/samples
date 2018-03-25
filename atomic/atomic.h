//
// Created by m_haid02 on 28.06.17.
//

#pragma once

#include <PACXX.h>
#include <stdatomic.h> 

using namespace pacxx::v2;

static int test_atomic(int argc, char *argv[]) {
  auto &exec = Executor::get(0);

  int value = 0;

  auto &da = exec.allocate<int>(1);

  da.upload(&value, 1);

  auto pa = da.get();

  auto vadd = [=](range &handle) {
	atomic_fetch_add((atomic_int*)pa, 1); 
  };

  exec.launch(vadd, {{16}, {16}});
  da.download(&value, 1);
  if (value == 16*16)
    return 0;

  return 1;
}
