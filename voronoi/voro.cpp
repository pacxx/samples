#define __PACXX_V2_NATIVE_RUNTIME

#include <gstorm.h>
#include <iomanip>
#include <iostream>
#include <range/v3/all.hpp>
#include <tuple>

#include <thrust/random.h>

using namespace ranges::v3;
using namespace gstorm;
using namespace pacxx::v2;

struct minFunctor {
  int m, n, k;

  minFunctor(int m, int n, int k) : m(m), n(n), k(k) {}

  int minVoro(int x_i, int y_i, int p, int q) const {
    if (q == m * n)
      return p;

    // coordinates of points p and q
    int y_q = q / m;
    int x_q = q - y_q * m;
    int y_p = p / m;
    int x_p = p - y_p * m;

    // squared distances
    int d_iq = (x_i - x_q) * (x_i - x_q) + (y_i - y_q) * (y_i - y_q);
    int d_ip = (x_i - x_p) * (x_i - x_p) + (y_i - y_p) * (y_i - y_p);

    if (d_iq < d_ip)
      return q; // q is closer
    else
      return p;
  }

  // For each point p+{-k,0,k}, we keep the Site with minimum distance
  template<typename T> int operator()(const T &t) const {
    // Current point and site
    int i = std::get<9>(t);
    int v = std::get<0>(t);

    // Current point coordinates
    int y = i / m;
    int x = i - y * m;

    if (x >= k) {
      v = minVoro(x, y, v, std::get<3>(t));

      if (y >= k)
        v = minVoro(x, y, v, std::get<8>(t));

      if (y + k < n)
        v = minVoro(x, y, v, std::get<7>(t));
    }

    if (x + m) {
      v = minVoro(x, y, v, std::get<1>(t));

      if (y >= k)
        v = minVoro(x, y, v, std::get<6>(t));
      if (y + k < n)
        v = minVoro(x, y, v, std::get<5>(t));
    }

    if (y >= k)
      v = minVoro(x, y, v, std::get<4>(t));
    if (y + k < n)
      v = minVoro(x, y, v, std::get<2>(t));

    // global return
    return v;
  }
};

void generate_random_sites(std::vector<int> &t, int Nb, int m, int n) {
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, m * n - 1);

  for (int k = 0; k < Nb; k++) {
    int index = dist(rng);
    t[index] = index + 1;
  }
}

// Export the tab to PGM image format
void vector_to_pgm(std::vector<int> &t, int m, int n, const char *out) {
  FILE *f;

  f = fopen(out, "w+t");
  fprintf(f, "P2\n");
  fprintf(f, "%d %d\n 253\n", m, n);

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      fprintf(f, "%d ",
              (int) (71 * t[j * m + i]) %
                  253); // Hash function to map values to [0,255]
    }
  }
  fprintf(f, "\n");
  fclose(f);
}

template<typename T> auto jfa(T &i, T &o, unsigned int k, int m, int n) {
  auto i0 = view::unbounded(i.begin());
  auto i1 = view::unbounded(i.begin() - k);
  auto i2 = view::unbounded(i.begin() - m * k);
  auto i3 = view::unbounded(i.begin() + k - m * k);
  auto i4 = view::unbounded(i.begin() - k + m * k);
  auto i5 = view::unbounded(i.begin() - k - m * k);
  auto zr =
      view::zip(i0, view::slice(i, k, n * m), view::slice(i, m * k, n * m), i1,
                i2, view::slice(i, k + m * k, n * m), i3, i4, i5,
                view::iota(0, n * m)) |
          view::take(n * m);

  gpu::algorithm::transform(zr, o, minFunctor(m, n, k));
}

int main(int argc, char *argv[]) {
  int m = 2048; // number of rows
  int n = 2048; // number of columns
  int s = 1000; // number of sites

  if (argc >= 2)
    m = std::stol(argv[1]);
  if (argc >= 3)
    n = std::stoi(argv[2]);

  int runs = 10;
  if (argc >= 4)
    runs = std::stoi(argv[3]);

#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
  Executor::Create<NativeRuntime>(0);
#endif

  // Host vector to encode a 2D image
  std::cout << "[Inititialize " << m << "x" << n << " Image]" << std::endl;
  std::vector<int> hseeds(m * n, m * n);
  generate_random_sites(hseeds, s, m, n);

  std::cout << "[Copy to Device]" << std::endl;

  auto temp = hseeds | gpu::copy;

  // JFA+1  : before entering the log(n) loop, we perform a jump with k=1
  std::cout << "[JFA stepping]" << std::endl;
  for (int i = 0; i < runs; ++i) {
    auto seeds = hseeds | gpu::copy;
    jfa(seeds, temp, 1, m, n);
    seeds.swap(temp);

    // JFA : main loop with k=n/2, n/4, ..., 1
    for (int k = std::max(m, n) / 2; k > 0; k /= 2) {
      jfa(seeds, temp, k, m, n);
      seeds.swap(temp);
    }
    hseeds = seeds;
    vector_to_pgm(hseeds, m, n, "discrete_voronoi.pgm");
  }
  return 0;
}
