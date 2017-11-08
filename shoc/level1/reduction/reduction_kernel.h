#ifndef REDUCTION_KERNEL_H_
#define REDUCTION_KERNEL_H_

#include <PACXX.h>

// The following class is a workaround for using dynamically sized
// shared memory in templated code. Without this workaround, the
// compiler would generate two shared memory arrays (one for SP
// and one for DP) of the same name and would generate an error.
template<class T>
class SharedMem {
public:
  inline T *getPointer() {
    [[pacxx::shared]] extern T s[];
    return s;
  };
};

// Specialization for double
template<>
class SharedMem<double> {
public:
  inline double *getPointer() {
    [[pacxx::shared]] extern double s_double[];
    return s_double;
  }
};

// specialization for float
template<>
class SharedMem<float> {
public:
  inline float *getPointer() {
    [[pacxx::shared]] extern float s_float[];
    return s_float;
  }
};

// Reduction Kernel
template<class T, int blockSize>
void
reduce(pacxx::v2::range &handle, const T *__restrict__ g_idata, T *__restrict__ g_odata,
       const unsigned int n) {

  const unsigned int tid = handle.get_local(0);
  unsigned int i = (handle.get_block(0) * (handle.get_block_size(0) * 2)) + tid;
  const unsigned int gridSize = handle.get_block_size(0) * 2 * handle.get_num_blocks(0);

  // Shared memory will be used for intrablock summation
  // NB: CC's < 1.3 seem incompatible with the templated dynamic
  // shared memory workaround.
  // Inspection with cuda-gdb shows sdata as a pointer to *global*
  // memory and incorrect results are obtained.  This explicit macro
  // works around this issue. Further, it is acceptable to specify
  // float, since CC's < 1.3 do not support double precision.

  SharedMem<T> sharedMem;
  volatile T *sdata = sharedMem.getPointer();

  sdata[tid] = 0.0f;

  // Reduce multiple elements per thread
  while (i < n) {
    sdata[tid] += g_idata[i] + g_idata[i + blockSize];
    i += gridSize;
  }
  handle.synchronize();

  // Reduce the contents of shared memory
  // NB: This is an unrolled loop, and assumes warp-syncrhonous
  // execution.
  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    handle.synchronize();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    handle.synchronize();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    handle.synchronize();
  }

  // NB2: This section would also need __sync calls if warp
  // synchronous execution were not assumed
  if (blockSize >= 64 && tid < 32)
    sdata[tid] += sdata[tid + 32];
  handle.synchronize();
  if (blockSize >= 32 && tid < 32)
    sdata[tid] += sdata[tid + 16];
  handle.synchronize();
  if (blockSize >= 16 && tid < 32)
    sdata[tid] += sdata[tid + 8];
  handle.synchronize();
  if (blockSize >= 8 && tid < 32)
    sdata[tid] += sdata[tid + 4];
  handle.synchronize();
  if (blockSize >= 4 && tid < 32)
    sdata[tid] += sdata[tid + 2];
  handle.synchronize();
  if (blockSize >= 2 && tid < 32)
    sdata[tid] += sdata[tid + 1];
  handle.synchronize();


  // Write result for this block to global memory
  if (tid == 0) {
    g_odata[handle.get_block(0)] = sdata[0];
  }
}

#endif // REDUCTION_KERNEL_H_
