#include <PACXX.h>
#include <gstorm.h>
#include <iomanip>
#include <random>
#include <range/v3/all.hpp>
#include <sstream>
#include <vector>

using namespace gstorm;
using namespace ranges::v3;
using namespace pacxx::v2;

#define __sq(x) ((x) * (x))
#define __cu(x) ((x) * (x) * (x))

// using data_t = typename pacxx::v2::vec<float, 4>::type;
using data_t = struct F4 {
  F4() : x(0), y(0), z(0), w(0) {}
  F4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
  float x, y, z, w;
};

int main(int argc, char *argv[]) {
  int runs = 10;
  size_t count = 1000;

  if (argc >= 2)
    count = std::stol(argv[1]);
  if (argc >= 3)
    runs = std::stoi(argv[2]);

#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
  Executor::Create<NativeRuntime>(0);
#endif

  std::vector<data_t> position(count), pos2(count), velocity(count);
  auto init = [](auto &pos) {
    std::mt19937 rng;

    std::uniform_real_distribution<float> rnd_pos(-1e11, 1e11);
    std::uniform_real_distribution<float> rnd_mass(1e22, 1e24);

    rng.seed(13122012);

    for (size_t i = 0; i != pos.size(); ++i) {
      pos[i].x = rnd_pos(rng);
      pos[i].y = rnd_pos(rng);
      pos[i].z = rnd_pos(rng);
      pos[i].w = rnd_mass(rng);
    }
  };

  init(position);

  constexpr auto G = -6.673e-11f;
  constexpr auto dt = 3600.f;
  constexpr auto eps2 = 0.00125f;

  auto nbody = [=](auto p, auto &v, auto &np, const auto &particles) {
    data_t a = {0.0f, 0.0f, 0.0f, 0.0f};
    data_t r = {0.0f, 0.0f, 0.0f, 0.0f};

    std::for_each(
        particles.begin(), particles.end(),
        [&](auto particle) { // use an unoptimized std::for_each for now
          r.x = p.x - particle.x;
          r.y = p.y - particle.y;
          r.z = p.z - particle.z;
          r.w = rsqrtf(__sq(r.x) + __sq(r.y) + __sq(r.z) + eps2);

          a.w = G * particle.w * __cu(r.w);

          a.x += a.w * r.x;
          a.y += a.w * r.y;
          a.z += a.w * r.z;
        });
    p.x += v.x * dt + a.x * 0.5f * __sq(dt);
    p.y += v.y * dt + a.y * 0.5f * __sq(dt);
    p.z += v.z * dt + a.z * 0.5f * __sq(dt);

    v.x += a.x * dt;
    v.y += a.y * dt;
    v.z += a.z * dt;

    np = p;
  };

  auto gpos = position | gpu::copy;
  auto gvelo = velocity | gpu::copy;
  auto gnpos = pos2 | gpu::copy;

  // gvectors have (currently) deleted copy constructores therefore we use a
  // gpu::ref which simulate a reference to a gvector
  auto gref = gpu::ref(gpos);
  auto pview = view::repeat(gref) |
               view::take(count); // we have to take count many referneces, if
                                  // we pass an infinite range to zip the
                                  // distance(range) will return 0 ... this is
                                  // strange?
  auto range = view::zip(gpos, gvelo, gnpos, pview);

  for (int i = 0; i < runs; ++i)
    gpu::algorithm::for_each(
        range, [&](auto &&tuple) { pacxx::meta::apply(nbody, tuple); });

  pos2 = gnpos;
  velocity = gvelo;
  std::stringstream ss;

  for (auto f : pos2)
    ss << std::fixed << std::setw(11) << std::setprecision(6) << f.x << " "
       << f.y << " " << f.z << " " << f.w << "\n";
  for (auto f : velocity)
    ss << std::fixed << std::setw(11) << std::setprecision(6) << f.x << " "
       << f.y << " " << f.z << " " << f.w << "\n";


  pacxx::common::write_string_to_file("pacxx.out", ss.str());
}
