#include <gstorm.h>
#include <iostream>
#include <range/v3/all.hpp>
#include <vector>
#include <detail/ranges/index.h>
#include <fstream>
#include <cmath>

#ifdef HAS_HELLO_WORLD
#warning "HELLO WORLD"
#endif

using namespace ranges;
using namespace gstorm;
using namespace pacxx::v2; 

struct pixel {
  unsigned char r, g, b, a;
//  unsigned int r : 8; // changed to a 32 bit field
//  unsigned int g : 8; // this results in a single 32 bit store
//  unsigned int b : 8; // not 4 8 bit stores for each value
//  unsigned int a : 8;

  pixel(int n) {
    r = (n & 63) << 2;
    g = (n << 3) & 255;
    b = (n >> 8) & 255;
    a = 255; 
  }
  pixel(unsigned char r = 0, unsigned char g = 0, unsigned char b = 0)
      : r(r), g(g), b(b) {}
};

std::ostream &operator<<(std::ostream &out, pixel p) {
  out << (int)p.r << " " << (int)p.g << " " << (int)p.b << "\n";
  return out;
}

template <typename Rng>
void writePPM(Rng& rng,  unsigned width, unsigned height,  const std::string &filename) {
  std::ofstream outputFile(filename.c_str());

  outputFile << "P3\n" << width << " " << height << "\n255\n";
  for (auto v : rng)
    outputFile << v; 
}

int main(int argc, char *argv[]) {

  unsigned width = 4096;
  unsigned height = 4096;
  unsigned niters = 100;

  if (argc >= 2)
    width = std::stol(argv[1]);
  if (argc >= 3)
    height = std::stoi(argv[2]);

  int runs = 10;
  if (argc >= 4)
    runs = std::stoi(argv[3]);

#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
  Executor::Create<NativeRuntime>(0);
  runs = 1;
#endif

  std::vector<pixel> data(width * height);
  auto gout = data | gpu::copy;

  auto idx2 = view::iota(0) | view::take(width * height);

  for (auto i = 0; i < runs; ++i)
    gpu::algorithm::transform(idx2, gout,
                              [=](const auto& idx) {
                                auto y = idx / width;
                                auto x = y ? idx % ( y * width) : idx;
                                float Zr = 0.0f;
                                float Zi = 0.0f;
                                float Cr = (y * (2.0f / height) - 1.5f);
                                float Ci = (x * (2.0f / width) - 1.0f);
                                int value = 0;
                                bool set = false;
                                for (unsigned i = 0; i < niters; i++) {
                                  const float ZiN = Zi * Zi;
                                  const float ZrN = Zr * Zr;
                                  if (ZiN + ZrN > 4.0f && !set) {
                                    value = i;
                                    set = true;
                                  }
                                  Zi *= Zr;
                                  Zi *= 2.0f;
                                  Zi += Ci;
                                  Zr = ZrN - ZiN + Cr;
                                }
                                return pixel(value);
                              });


  data = gout;
  writePPM(data, width, height, "mandel.ppm");
}
