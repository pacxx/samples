#include <PACXX.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Utility.h"

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op) {
  ;
}

// ****************************************************************************
// Function: triad
//
// Purpose:
//   A simple vector addition kernel
//   C = A + s*B
//
// Arguments:
//   A,B - input vectors
//   C - output vectors
//   s - scalar
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
template<typename CFG>
void triad(CFG &handle, float *A, float *B, float *C, float s) {
  int gid = handle.get_global(0);
  C[gid] = A[gid] + s * B[gid];
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Implements the Stream Triad benchmark in CUDA.  This benchmark
//   is designed to test CUDA's overall data transfer speed. It executes
//   a vector addition operation with no temporal reuse. Data is read
//   directly from the global memory. This implementation tiles the input
//   array and pipelines the vector addition computation with
//   the data download for the next tile. However, since data transfer from
//   host to device is much more expensive than the simple vector computation,
//   data transfer operations should completely dominate the execution time.
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser (contains input parameters)
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
  const bool verbose = op.getOptionBool("verbose");
  const int n_passes = op.getOptionInt("passes");

  // 256k through 8M bytes
  const int nSizes = 1;
  const size_t blockSizes[] = {16384};
  const size_t memSize = 16384;
  const size_t numMaxFloats = 1024 * memSize;
  const size_t halfNumFloats = numMaxFloats / 2;

  // Create some host memory pattern
  srand48(8650341L);

  std::vector<float> h_mem(numMaxFloats);

  auto &exec = pacxx::v2::Executor::get(0);

  // Allocate some device memory
  auto &memA0 = exec.allocate<float>(blockSizes[nSizes - 1] * 1024 / sizeof(float));
  auto &memB0 = exec.allocate<float>(blockSizes[nSizes - 1] * 1024 / sizeof(float));
  auto &memC0 = exec.allocate<float>(blockSizes[nSizes - 1] * 1024 / sizeof(float));

  auto d_memA0 = memA0.get();
  auto d_memB0 = memB0.get();
  auto d_memC0 = memC0.get();

  auto &memA1 = exec.allocate<float>(blockSizes[nSizes - 1] * 1024 / sizeof(float));
  auto &memB1 = exec.allocate<float>(blockSizes[nSizes - 1] * 1024 / sizeof(float));
  auto &memC1 = exec.allocate<float>(blockSizes[nSizes - 1] * 1024 / sizeof(float));

  auto d_memA1 = memA1.get();
  auto d_memB1 = memB1.get();
  auto d_memC1 = memC1.get();

  float scalar = 1.75f;

  const size_t blockSize = 128;

  // Number of passes. Use a large number for stress testing.
  // A small value is sufficient for computing sustained performance.
  char sizeStr[256];
  for (int pass = 0; pass < n_passes; ++pass) {
    // Step through sizes forward
    for (int i = 0; i < nSizes; ++i) {
      int elemsInBlock = blockSizes[i] * 1024 / sizeof(float);
      for (int j = 0; j < halfNumFloats; ++j)
        h_mem[j] = h_mem[halfNumFloats + j]
            = (float) (drand48() * 10.0);

      // Copy input memory to the device
      if (verbose)
        cout << ">> Executing Triad with vectors of length "
             << numMaxFloats << " and block size of "
             << elemsInBlock << " elements." << "\n";
      sprintf(sizeStr, "Block:%05ldKB", blockSizes[i]);

      // start submitting blocks of data of size elemsInBlock
      // overlap the computation of one block with the data
      // download for the next block and the results upload for
      // the previous block
      int crtIdx = 0;
      size_t globalWorkSize = elemsInBlock / blockSize;

      auto &event = exec.createEvent();

      event.start();
      // TODO: we need streams
      memA0.upload(h_mem.data(), blockSizes[i] * 1024 / sizeof(float));
      memB0.upload(h_mem.data(), blockSizes[i] * 1024 / sizeof(float));

      exec.launch([=](auto &handle) {
        triad(handle, d_memA0, d_memB0, d_memC0, scalar);
      }, {{globalWorkSize}, {blockSize}});

      if (elemsInBlock < numMaxFloats) {
        memA1.upload(h_mem.data() + elemsInBlock, blockSizes[i] * 1024 / sizeof(float));
        memB1.upload(h_mem.data() + elemsInBlock, blockSizes[i] * 1024 / sizeof(float));
      }

      int blockIdx = 1;
      unsigned int currStream = 1;
      while (crtIdx < numMaxFloats) {
        currStream = blockIdx & 1;
        // Start copying back the answer from the last kernel
        if (currStream) {
          memC0.download(h_mem.data() + crtIdx, elemsInBlock);
        } else {
          memC1.download(h_mem.data() + crtIdx, elemsInBlock);
        }

        crtIdx += elemsInBlock;

        if (crtIdx < numMaxFloats) {
          // Execute the kernel
          if (currStream) {
            exec.launch([=](auto &handle) {
              triad(handle, d_memA1, d_memB1, d_memC1, scalar);
            }, {{globalWorkSize}, {blockSize}});
          } else {
            exec.launch([=](auto &handle) {
              triad(handle, d_memA0, d_memB0, d_memC0, scalar);
            }, {{globalWorkSize}, {blockSize}});
          }
        }

        if (crtIdx + elemsInBlock < numMaxFloats) {
          // Download data for next block
          if (currStream) {
            memA0.upload(h_mem.data() + crtIdx + elemsInBlock, blockSizes[i] * 1024 / sizeof(float));
            memB0.upload(h_mem.data() + crtIdx + elemsInBlock, blockSizes[i] * 1024 / sizeof(float));
          } else {
            memA1.upload(h_mem.data() + crtIdx + elemsInBlock, blockSizes[i] * 1024 / sizeof(float));
            memB1.upload(h_mem.data() + crtIdx + elemsInBlock, blockSizes[i] * 1024 / sizeof(float));
          }
        }
        blockIdx += 1;
        currStream = !currStream;
      }

      exec.synchronize();
      event.stop();
      double time = event.result();

      double triad = ((double) numMaxFloats * 2.0) / (time * 1e9);
      resultDB.AddResult("TriadFlops", sizeStr, "GFLOP/s", triad);

      double bdwth = ((double) numMaxFloats * sizeof(float) * 3.0)
          / (time * 1000. * 1000. * 1000.);
      resultDB.AddResult("TriadBdwth", sizeStr, "GB/s", bdwth);

      // Checking memory for correctness. The two halves of the array
      // should have the same results.
      if (verbose)
        cout << ">> checking memory\n";
      for (int j = 0; j < halfNumFloats; ++j) {
        if (h_mem[j] != h_mem[j + halfNumFloats]) {
          cout << "Error; hostMem[" << j << "]=" << h_mem[j]
               << " is different from its twin element hostMem["
               << (j + halfNumFloats) << "]: "
               << h_mem[j + halfNumFloats] << "stopping check\n";
          break;
        }
      }
      if (verbose)
        cout << ">> finish!" << endl;

      // Zero out the test host memory
      for (int j = 0; j < numMaxFloats; ++j)
        h_mem[j] = 0.0f;
    }
  }
}
