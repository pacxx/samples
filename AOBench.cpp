//modified by huihuisun for PACXX 2018.02.19
// -*- mode: c++ -*-
// ======================================================================== //
// Copyright 2017 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

/* originally imported from https://github.com/ispc/ispc, under the 
   following license */
/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:
  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
  * Neither the name of Intel Corporation nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
  OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/
/*
  Based on Syoyo Fujita's aobench: http://code.google.com/p/aobench
*/

/*! special thanks to
  - Syoyo Fujita, who apparently created the first aobench that was the root of all this!
  - Matt Pharr, who's aoBench variant this is based on
  - the OSPRay Project, and in particular Johannes Guenther, for the random number generator
 */
#include <PACXX.h>
//#include <pacxx/detail/device/DevicePrintf.h>
#include <chrono>
#include <ctime>

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#pragma warning (disable: 4244)
#pragma warning (disable: 4305)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#ifdef __linux__
#include <malloc.h>
#endif
#include <math.h>
#include <map>
#include <string>
#include <algorithm>
#include <sys/types.h>
using namespace pacxx::v2;
namespace{
#define NAO_SAMPLES		8
#define M_PI 3.1415926535f
/* random number generator, taken from the ospray project, http://www.ospray.org 
   Special thanks to Johannes Guenther who originally added this neat
   rng to ospray!
 */
#define TABLE_SIZE 32
#define WARMUP_ITERATIONS 7
struct RNGState {
  int seed;
  int state;
  int table[TABLE_SIZE];
};

void rng_seed(struct RNGState *rng, int s)
{
  const int a = 16807; 
  const int m = 2147483647;
  const int q = 127773;
  const int r = 2836;
  
  if (s == 0) rng->seed = 1;
  else rng->seed = s & 0x7FFFFFFF;
  
  for (int j = TABLE_SIZE+WARMUP_ITERATIONS; j >= 0; j--) {
    int k = rng->seed / q;
    rng->seed = a*(rng->seed - k*q) - r*k;
    rng->seed = rng->seed & 0x7FFFFFFF;
    if (j < TABLE_SIZE) rng->table[j] = rng->seed;
  }
  rng->state = rng->table[0];
}

float rng_getInt(struct RNGState *rng)
{
  const int a = 16807;
  const int m = 2147483647;
  const int q = 127773;
  const int r = 2836;
  const int f = 1 + (2147483647 / TABLE_SIZE);
  
  int k = rng->seed / q;
  rng->seed = a*(rng->seed - k*q) - r*k;
  rng->seed = rng->seed & 0x7FFFFFFF;
  int j = fminf(rng->state / f, TABLE_SIZE-1);
  rng->state = rng->table[j];
  rng->table[j] = rng->seed;
  return rng->state;
}

float rng_getFloat(struct RNGState *rng)
{
  return rng_getInt(rng) / 2147483647.0f;
}

struct vec3f {
  float x,y,z;
};

struct Isect {
  float      t;
  struct vec3f p;
  struct vec3f n;
  int        hit; 
};

struct Sphere {
  struct vec3f center;
  float      radius;
};

struct Plane {
  struct vec3f    p;
  struct vec3f    n;
};

struct Ray {
  struct vec3f org;
  struct vec3f dir;
};


inline float dot3f(struct vec3f a, struct vec3f b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline struct vec3f cross3f(struct vec3f v0, struct vec3f v1) {
  struct vec3f ret;
  ret.x = v0.y * v1.z - v0.z * v1.y;
  ret.y = v0.z * v1.x - v0.x * v1.z;
  ret.z = v0.x * v1.y - v0.y * v1.x;
  return ret;
}

inline struct vec3f mul3ff(struct vec3f v, float f)
{
  struct vec3f ret;
  ret.x = v.x * f;
  ret.y = v.y * f;
  ret.z = v.z * f;
  return ret;
}

inline struct vec3f add3f (struct vec3f a, struct vec3f b)
{
  struct vec3f ret;
  ret.x = a.x+b.x;
  ret.y = a.y+b.y;
  ret.z = a.z+b.z;
  return ret;
}

inline struct vec3f sub3f (struct vec3f a, struct vec3f b)
{
  struct vec3f ret;
  ret.x = a.x-b.x;
  ret.y = a.y-b.y;
  ret.z = a.z-b.z;
  return ret;
}

inline struct vec3f madd3ff(struct vec3f a, float f, struct vec3f b)
{
  struct vec3f ret;
  ret.x = a.x + f * b.x;
  ret.y = a.y + f * b.y;
  ret.z = a.z + f * b.z;
  return ret;
}

//float rsqrt(float f);
//float absf(float f);
inline struct vec3f normalize3f(struct vec3f v)
{
  float len2 = dot3f(v, v);
  float invLen = rsqrtf(len2);
  return mul3ff(v,invLen);
}

inline void ray_plane_intersect(struct Isect *isect, struct Ray ray, struct Plane plane)
{
  float d = -dot3f(plane.p, plane.n);
  float v =  dot3f(ray.dir, plane.n);

  if (fabs(v) < 1.0e-17f) 
    return;
  else {
    float t = -(dot3f(ray.org, plane.n) + d) / v;
    
    if ((t > 0.0f) && (t < isect->t)) {
      isect->t = t;
      isect->hit = 1;
      isect->p = madd3ff(ray.org,t,ray.dir);
      isect->n = plane.n;
    }
  }
}


inline void ray_sphere_intersect(struct Isect *isect, struct Ray ray, struct Sphere sphere)
{
  struct vec3f rs = sub3f(ray.org,sphere.center);
  
  float B = dot3f(rs, ray.dir);
  float C = dot3f(rs, rs) - sphere.radius * sphere.radius;
  float D = B * B - C;
  
  if (D > 0.f) {
    float t = -B - sqrtf(D);

   if ((t > 0.0f) && (t < isect->t)) {
      isect->t = t;
      isect->hit = 1;
      isect->p = madd3ff(ray.org,t,ray.dir);
      isect->n = normalize3f(sub3f(isect->p, sphere.center));
    }
  }
}


inline void orthoBasis(struct vec3f basis[3], struct vec3f n)
{
  basis[2] = n;
  basis[1].x = 0.0f;
  basis[1].y = 0.0f;
  basis[1].z = 0.0f;

  if ((n.x < 0.6f) && (n.x > -0.6f)) {
    basis[1].x = 1.0f;
  } else if ((n.y < 0.6f) && (n.y > -0.6f)) {
    basis[1].y = 1.0f;
  } else if ((n.z < 0.6f) && (n.z > -0.6f)) {
    basis[1].z = 1.0f;
  } else {
    basis[1].x = 1.0f;
  }

  basis[0] = normalize3f(cross3f(basis[1], basis[2]));
  basis[1] = normalize3f(cross3f(basis[2], basis[0]));
}


float ambient_occlusion(struct Isect *isect, struct Plane plane, struct Sphere spheres[3],
                        struct RNGState *rngstate) {
  float eps = 0.0001f;
  struct vec3f p, n;
  struct vec3f basis[3];
  float occlusion = 0.0f;

  p = madd3ff(isect->p,eps,isect->n);

  orthoBasis(basis, isect->n);

  const int ntheta = NAO_SAMPLES;
  const int nphi   = NAO_SAMPLES;
  for (int j = 0; j < ntheta; j++) {
    for (int i = 0; i < nphi; i++) {
      struct Ray ray;
      struct Isect occIsect;

      float theta = sqrtf(rng_getFloat(rngstate));
      float phi   = 2.0f * M_PI * rng_getFloat(rngstate);
      float x = cosf(phi) * theta;
      float y = sinf(phi) * theta;
      float z = sqrtf(1.0f - theta * theta);

      // local . global
      float rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
      float ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
      float rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

      ray.org = p;
      ray.dir.x = rx;
      ray.dir.y = ry;
      ray.dir.z = rz;

      occIsect.t   = 1.0e+17f;
      occIsect.hit = 0;

      for (int snum = 0; snum < 3; ++snum)
        ray_sphere_intersect(&occIsect, ray, spheres[snum]); 
      ray_plane_intersect (&occIsect, ray, plane); 

      if (occIsect.hit) occlusion += 1.0f;
    }
  }

  occlusion = (ntheta * nphi - occlusion) / (float)(ntheta * nphi);
  return occlusion;
}


/* Compute the image for the scanlines from [y0,y1), for an overall image
   of width w and height h.
*/

struct Image {
  Image(size_t width, size_t height)
    : width(width), height(height),
      pix(new float[3*width*height])
  {};
  ~Image() { delete[] pix; }

  size_t width, height;
  float *pix;

  void savePPM(const char *fName);
};

void Image::savePPM(const char *fname)
{ 
  char *tmp = new char[3*width*height];
  for (int i=0;i<3*width*height;i++) {
    tmp[i] = int(std::max(0,std::min(255,int(pix[i]*256.f))));
  }
       
  FILE *fp = fopen(fname, "wb");
  if (!fp) {
    perror(fname);
    exit(1);
  }
  
  fprintf(fp, "P6\n");
  fprintf(fp, "%ld %ld\n", width, height);
  fprintf(fp, "255\n");
  fwrite(tmp, 3*width*height, 1, fp);
  fclose(fp);
  printf("Wrote image file %s\n", fname);
}
}

int main(int argc, char **argv)
{
  size_t width = 600, height = 800;
  int numSubSamples = 4;
  Image image(width,height);

  auto &exec = Executor::get(0);

  int size = 3*width*height;


  std::chrono::time_point<std::chrono::system_clock> clock_begin
    = std::chrono::system_clock::now();

  //upload  
  auto &imagetemp = exec.allocate<float>(size);
  imagetemp.upload(image.pix, size);
  auto pimage = imagetemp.get();

  //kernel
  /* Compute the image for the scanlines from [y0,y1), for an overall image
   of width w and height h.
*/
  auto aoBench = [=](range &config) {
    struct Plane plane = { { 0.0f, -0.5f, 0.0f }, { 0.f, 1.f, 0.f } };
  struct Sphere spheres[3] = {
    { { -2.0f, 0.0f, -3.5f }, 0.5f },
    { { -0.5f, 0.0f, -3.0f }, 0.5f },
    { { 1.0f, 0.0f, -2.2f }, 0.5f } };
  struct RNGState rngstate;
  
  float invSamples = 1.f / numSubSamples;
  
  int x = config.get_local(0);
  int y = config.get_block(0);
  int offset = 3 * (y * width + x);
  rng_seed(&rngstate,offset); //, programIndex + (y0 << (programIndex & 15)));
  
  float ret = 0.f;
  for (int v=0;v<numSubSamples;v++)
   for (int u=0;u<numSubSamples;u++) {
      float du = (float)u * invSamples, dv = (float)v * invSamples;
      
      // Figure out x,y pixel in NDC
      float px =  (x + du - (width / 2.0f)) / (width / 2.0f);
      float py = -(y + dv - (height / 2.0f)) / (height / 2.0f);
      struct Ray ray;
      struct Isect isect;
      
      ray.org.x = 0.f;
      ray.org.y = 0.f;
      ray.org.z = 0.f;
      
      // Poor man's perspective projection
      ray.dir.x = px;
      ray.dir.y = py;
      ray.dir.z = -1.0f;
      ray.dir = normalize3f(ray.dir);
      
      isect.t   = 1.0e+17f;
      isect.hit = 0;
      
      for (int snum = 0; snum < 3; ++snum)
         ray_sphere_intersect(&isect, ray, spheres[snum]);
     ray_plane_intersect(&isect, ray, plane);
     if (isect.hit) {
       ret += ambient_occlusion(&isect, plane, spheres, &rngstate);       
      }
   }
  ret *= (invSamples * invSamples);
  pimage[offset] = ret;
  pimage[offset+1] = ret;
  pimage[offset+2] = ret;
};

  exec.launch(aoBench, {{height, 1, 1}, {width, 1, 1}});

  //download
  imagetemp.download(image.pix, size);


  std::chrono::time_point<std::chrono::system_clock> clock_end
    = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = clock_end-clock_begin;

  // Report results and save image
  std::cout << "[aobench cl]:\t" << seconds.count() << "s "
            << ", for a " << width << "x" << height << " pixels" << std::endl;
  image.savePPM("ao-cl.ppm");

  return 0;
}
