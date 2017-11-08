
#include <vector>
#include <random>
#include <iterator>
#include <PACXX.h>
#include <type_traits>
#include <typeinfo>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace pacxx::v2;
using namespace std;

// this is a workaround
// FIXME: pacxx kernel lambdas should not be allowed to capture
// non-const global variables!
const unsigned width = 4096;
const unsigned threads = 1024;
const unsigned matrix_size = width * width;

void initMatrix(vector<int> & matrix) {
  random_device rnd_device;
  mt19937 mersenne_engine(rnd_device());
  uniform_int_distribution<int> dist(1, 52);
  auto gen = bind(dist, mersenne_engine);
  generate(begin(matrix), end(matrix), gen);
}

void clearMatrix(int* matrix) {
  for(unsigned i = 0; i < width; ++i)
    for(unsigned j=0; j < width; ++j)
      matrix[i * width + j] = 0;
}

void calcSeq(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
  for (unsigned Col = 0; Col < width; ++Col)
    for (unsigned Row = 0; Row < width; ++Row) {
      float sum = 0;
      for (unsigned k = 0; k < width; ++k) {
        sum += a[Row * width + k] * b[k * width + Col];
      }
      c[Row * width + Col] = sum;
    }
}

bool compareMatrices(const std::vector<int> &first, const std::vector<int> &second) {
  bool equal = true;
  for(unsigned i = 0; i < width ; ++i)
    for(unsigned j = 0; j < width; ++j)
      if(first[i * width +j] - second[i * width +j] != 0)
        equal = false;
  return equal;
}

void calcPACXX(int* a, int* b, int* c) {
#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
  Executor::Create<NativeRuntime>(0);
#endif

  auto& exec = Executor::get(0);

  auto& dev_a = exec.allocate<int>(matrix_size, a);
  auto& dev_b = exec.allocate<int>(matrix_size, b);
  auto& dev_c = exec.allocate<int>(matrix_size, c);

  auto pa = dev_a.get();
  auto pb = dev_b.get();
  auto pc = dev_c.get();

  auto test = [=](auto &config) {
    auto row = config.get_global(0);
    auto column = config.get_global(1);
    int val = 0;
    for (unsigned i = 0; i < width; ++i)
      val += pa[row * width + i] * pb[i * width + column];
    pc[row * width + column] = val;
  };

  exec.launch(test, {{width / threads, width}, {threads, 1}, 0});
  exec.synchronize();
}

int main(int argc, char **argv) {

  std::vector<int> a(matrix_size), b(matrix_size), c(matrix_size), c_host(matrix_size);

  initMatrix(a);

  initMatrix(b);

  clearMatrix(c.data());

  calcPACXX(a.data(), b.data(), c.data());

  calcSeq(a, b, c_host);

  std::cout << compareMatrices(c, c_host) << std::endl;

  return 0;
}
