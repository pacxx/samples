#include <PACXX.h>
#include <algorithm>
#include <vector>

using namespace pacxx::v2;

#define SIZE 2000

struct Heap {
  int *tasks;
  int head;
};

int master(Heap &heap, int x) {
  if (x <= 1)
    return 1;
  else {
    //      Task *t = newTask(heap , x-1);
    int a = master(heap, x - 2);
    int b = master(heap, x - 1); // fetchResult(heap , t);

    return a + b;
  }
}

int main(int argc, char *argv[]) {

  Executor::Create<CUDARuntime>(0); // create an executor
  auto &exec = Executor::get(0);    // retrieve the default executor

  size_t size = 8;

  std::vector<int> a(size);
  auto &da = exec.allocate<int>(a.size()); // allocate some memory on the device
  da.upload(a.data(), a.size());           // upload data to the device
  auto pa = da.get(); // grab the raw pointer from the device address space

  auto &taskMem =
      exec.allocate<int>(SIZE); // allocate some memory on the device
  auto tasks =
      taskMem.get(); // grab the raw pointer from the device address space

  auto vadd = [=](range &config) { // define the vector addition kernel
    auto i = config.get_global(
        0); // get the global id (in x-dimension) for the thread

    Heap heap;
    heap.tasks = tasks;

    if (i == 0) {
      pa[i] = master(heap, 13);
    } else
      pa[i] = 0;
  };

  exec.launch(vadd,
              {{1}, {size}}); // launch the kernel with 128 threads in 1 block
  da.download(a.data(), a.size()); // download the results from the device

  for (int i = 0; i < size; i++)
    std::cout << a[i] << std::endl;

  return 1;
}
