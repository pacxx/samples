#include <PACXX.h>
#include <algorithm>
#include <gstorm.h>
#include <iostream>
#include <range/v3/all.hpp>
#include <thrust/random.h>
#include <vector>

using namespace pacxx::v2;

template <typename T> void printRng(const T &Rng) {
  std::for_each(Rng.begin(), Rng.end(), [](auto v) { std::cout << v; });
  std::cout << std::endl;
}

struct estimate_pi {
  float operator()(unsigned int thread_id) {
    float sum = 0;
    unsigned int N = 5000; // samples per stream

    // note that M * N <= default_random_engine::max,
    // which is also the period of this particular RNG
    // this ensures the substreams are disjoint

    // create a random number generator
    // note that each thread uses an RNG with the same seed
    thrust::default_random_engine rng;

    // jump past the numbers used by the subsequences before me
    rng.discard(N * thread_id);

    // create a mapping from random numbers to [0,1)
    thrust::uniform_real_distribution<float> u01(0, 1);

    // take N samples in a quarter circle
    for (unsigned int i = 0; i < N; ++i) {
      // draw a sample from the unit square
      float x = u01(rng);
      float y = u01(rng);

      // measure distance from the origin
      float dist = sqrtf(x * x + y * y);

      // add 1.0f if (u0,u1) is inside the quarter circle
      if (dist <= 1.0f)
        sum += 1.0f;
    }

    // multiply by 4 to get the area of the whole circle
    sum *= 4.0f;

    // divide by N
    return sum / N;
  }
};

using namespace gstorm;
using namespace ranges::v3;

int main(int argc, char *argv[]) {
  size_t M = 30000;
  int runs = 10; 
  if (argc >= 2)
    M = std::stol(argv[1]);

  if (argc >= 3)
    runs = std::stoi(argv[2]);

  auto estimate = 0.0f;
  for (int i = 0; i < runs; ++i)
    estimate = gpu::algorithm::reduce(
        view::iota(0) | view::take(M) | view::transform(estimate_pi()), 0.0f,
        [](auto &&lhs, auto &&rhs) { return lhs + rhs; });
  estimate /= M;
  std::cout << "pi is around " << estimate << std::endl;
}
