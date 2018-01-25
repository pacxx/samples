//
// Created by m_haid02 on 16.06.17.
//

#include <PACXX.h>
using namespace pacxx::v2;

const float fR = 0.2126f;
const float fG = 0.7152f;
const float fB = 0.0722f;

struct pixel_char {
  unsigned char r, g, b, a;

  pixel_char(int n) {
    r = (n & 63) << 2;
    g = (n << 3) & 255;
    b = (n >> 8) & 255;
    a = 255;
  }
  pixel_char(unsigned char r = 0, unsigned char g = 0, unsigned char b = 0)
      : r(r), g(g), b(b) {}
};

struct pixel_bit {
  unsigned int r : 8; // changed to a 32 bit field
  unsigned int g : 8; // this results in a single 32 bit store
  unsigned int b : 8; // not 4 8 bit stores for each value
  unsigned int a : 8;

  pixel_bit(int n) {
    r = (n & 63) << 2;
    g = (n << 3) & 255;
    b = (n >> 8) & 255;
    a = 255;
  }
  pixel_bit(unsigned char r = 0, unsigned char g = 0, unsigned char b = 0)
      : r(r), g(g), b(b) {}
};

struct pixel_vec {
  typename pacxx::v2::vec<unsigned char, 4>::type vec;

  pixel_vec(int n) {
    vec.x = (n & 63) << 2;
    vec.y = (n << 3) & 255;
    vec.z = (n >> 8) & 255;
    vec.w = 255;
  }
  pixel_vec(unsigned char r = 0, unsigned char g = 0, unsigned char b = 0) {
    vec.x = r;
    vec.y = g;
    vec.z = b;
    vec.w = 0;
  }
};

int main() {

#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
  Executor::Create<NativeRuntime>(0);
#endif

  auto &exec = Executor::get(0);

  size_t size = 1 << 24;

  std::vector<pixel_char> a(size), a2(size);
  std::vector<pixel_bit> b(size), b2(size);
  std::vector<pixel_vec> c(size), c2(size);

  auto &da = exec.allocate<pixel_char>(a.size());
  da.upload(a.data(), a.size());
  auto pa = da.get();

  auto &da2 = exec.allocate<pixel_char>(a2.size());
  da2.upload(a2.data(), a2.size());
  auto pa2 = da2.get();

  auto &db = exec.allocate<pixel_bit>(b.size());
  db.upload(b.data(), b.size());
  auto pb = db.get();

  auto &db2 = exec.allocate<pixel_bit>(b2.size());
  db2.upload(b2.data(), b2.size());
  auto pb2 = db2.get();

  auto &dc = exec.allocate<pixel_vec>(c.size());
  dc.upload(c.data(), c.size());
  auto pc = dc.get();

  auto &dc2 = exec.allocate<pixel_vec>(c2.size());
  dc2.upload(c2.data(), c2.size());
  auto pc2 = dc2.get();

  auto vadd_char = [=](range &config) {
    auto i = config.get_global(0);
    if (i < size) {
      auto y = fR * pa2[i].r + fB * pa2[i].b + fG * pa2[i].g;
      pa[i] = pixel_char(y, y, y);
    }
  };

  auto vadd_bit = [=](range &config) {
    auto i = config.get_global(0);
    if (i < size) {
      auto y = fR * pb2[i].r + fB * pb2[i].b + fG * pb2[i].g;
      pb[i] = pixel_bit(y, y, y);
    }
  };

  auto vadd_vec = [=](range &config) {
    auto i = config.get_global(0);
    if (i < size) {
      auto y = fR * pc2[i].vec.x + fB * pc2[i].vec.z + fG * pc2[i].vec.y;
      pc[i] = pixel_vec(y, y, y);
    }
  };

  for (auto i = 0; i < 100; ++i) {
    exec.launch(vadd_char, {{size / 128}, {128}, exec.getID()});
    exec.launch(vadd_bit, {{size / 128}, {128}, exec.getID()});
    exec.launch(vadd_vec, {{size / 128}, {128}, exec.getID()});
  }
  da.download(a.data(), a.size());
  db.download(b.data(), b.size());
}
