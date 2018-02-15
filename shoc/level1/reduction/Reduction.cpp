#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include "reduction_kernel.h"
#include "OptionParser.h"
#include "ResultDatabase.h"

#include <PACXX.h>

using namespace std;

template<class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

// ****************************************************************************
// Function: reduceCPU
//
// Purpose:
//   Simple cpu reduce routine to verify device results
//
// Arguments:
//   data : the input data
//   size : size of the input data
//
// Returns:  sum of the data
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
template<class T>
T reduceCPU(const T *data, int size) {
  T sum = 0;
  for (int i = 0; i < size; i++) {
    sum += data[i];
  }
  return sum;
}

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
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void
addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("iterations", OPT_INT, "256",
               "specify reduction iterations");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Driver for the reduction benchmark.  Detects double precision capability
//   and calls the RunTest function appropriately
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {

  auto &exec = pacxx::v2::Executor::get(0);

  cout << "Running single precision test" << endl;
  RunTest<float>("Reduction", resultDB, op);

  if (exec.supportsDoublePrecission()) {
    cout << "Running double precision test" << endl;
    RunTest<double>("Reduction-DP", resultDB, op);
  } else {
    cout << "Skipping double precision test" << endl;
    char atts[1024] = "DP_Not_Supported";
    // resultDB requires neg entry for every possible result
    int passes = op.getOptionInt("passes");
    for (int k = 0; k < passes; k++) {
      resultDB.AddResult("Reduction-DP", atts, "GB/s", FLT_MAX);
      resultDB.AddResult("Reduction-DP_PCIe", atts, "GB/s", FLT_MAX);
      resultDB.AddResult("Reduction-DP_Parity", atts, "GB/s", FLT_MAX);
    }
  }

}
// ****************************************************************************
// Function: RunTest
//
// Purpose:
//   Primary method for the reduction benchmark
//
// Arguments:
//   testName: the name of the test currently being executed (specifying SP or
//             DP)
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
template<class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op) {
  int prob_sizes[5] = {1, 8, 16, 32, 64};

  int size = prob_sizes[op.getOptionInt("size") - 1];
  size = (size * 1024 * 1024);

  auto &exec = pacxx::v2::Executor::get(0);
  std::vector<T> h_idata(size);

  // Initialize host memory
  cout << "Initializing host memory." << endl;
  for (int i = 0; i < size; i++) {
    h_idata[i] = i % 3; //Fill with some pattern
  }

  // allocate device memory

  auto &d_idata = exec.allocate<T>(size);

  int num_threads = 256; // NB: Update template to kernel launch
  // if this is changed
  int num_blocks = 64;
  int smem_size = sizeof(T) * num_threads;
  // allocate mem for the result on host side
  std::vector<T> h_odata(num_blocks);
  auto &d_odata = exec.allocate<T>(num_blocks);

  int passes = op.getOptionInt("passes");
  int iters = op.getOptionInt("iterations");

  cout << "Running benchmark." << endl;
  for (int k = 0; k < passes; k++) {
    // Copy data to GPU
    auto event = exec.createEvent();
    event->start();
    d_idata.upload(h_idata.data(), h_idata.size());
    exec.synchronize();
    event->stop();

    // Get elapsed time
    float transfer_time = event->result();

    // Execute kernel
    event->start();

    auto pd_idata = d_idata.get();
    auto pd_odata = d_odata.get();

    for (int m = 0; m < iters; m++) {
      exec.launch([=](auto &handle) {
        reduce<T, 256>(handle, pd_idata, pd_odata, size);
      }, {{static_cast<size_t>(num_blocks)}, {static_cast<size_t>(num_threads)}, static_cast<unsigned int>(smem_size)});
    }
    exec.synchronize();
    event->stop();

    // Get kernel time
    float totalReduceTime = event->result();
    double avg_time = totalReduceTime / (double) iters;

    // Copy back to host
    event->start();
    d_odata.download(h_odata.data(), h_odata.size());
    exec.synchronize();
    event->stop();
    float output_time = event->result();
    transfer_time += output_time;

    T dev_result = 0;
    for (int i = 0; i < num_blocks; i++) {
      dev_result += h_odata[i];
    }

    // compute reference solution
    T cpu_result = reduceCPU<T>(h_idata.data(), h_idata.size());
    double threshold = 1.0e-6;
    T diff = fabs(dev_result - cpu_result);

    cout << "Test ";
    if (diff < threshold)
      cout << "Passed";
    else {
      cout << "FAILED\n";
      cout << "Diff: " << diff;
      return; // (don't report erroneous results)
    }
    cout << endl;

    // Calculate results
    char atts[1024];
    sprintf(atts, "%d_items", size);
    double gbytes = (double) (size * sizeof(T)) / (1000. * 1000. * 1000.);
    resultDB.AddResult(testName, atts, "GB/s", gbytes / avg_time);
    resultDB.AddResult(testName + "_PCIe", atts, "GB/s", gbytes /
        (avg_time + transfer_time));
    resultDB.AddResult(testName + "_Parity", atts, "N",
                       transfer_time / avg_time);
  }
}
