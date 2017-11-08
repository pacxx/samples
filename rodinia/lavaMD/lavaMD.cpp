//	2009.12 Lukasz G. Szafaryn
//		-- entire code written


#include <PACXX.h>
#include <pacxx/detail/device/DeviceCode.h>
using namespace pacxx::v2;
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define fp float

#define NUMBER_PAR_PER_BOX 100          // keep this low to allow more blocks that share shared memory to
// run concurrently, code does not work for larger than 110, more
// speedup can be achieved with larger number and no shared memory used

#define NUMBER_THREADS 128            // this should be roughly equal to NUMBER_PAR_PER_BOX for best performance

typedef struct {
  fp x, y, z;

} THREE_VECTOR;

typedef struct {
  fp x, y, z, w;

} FOUR_VECTOR;

//typedef typename pacxx::v2::vec<fp, 4>::type FOUR_VECTOR;

typedef struct nei_str {
  // neighbor box
  int x, y, z;
  int number;
  long offset;

} nei_str;

typedef struct box_str {
  // home box
  int x, y, z;
  int number;
  long offset;

  // neighbor boxes
  int nn;
  nei_str nei[26];

} box_str;

typedef struct par_str {
  fp alpha;
} par_str;

typedef struct dim_str {
  // input arguments
  int cur_arg;
  int arch_arg;
  int cores_arg;
  int boxes1d_arg;

  // system memory
  long number_boxes;
  long box_mem;
  long space_elem;
  long space_mem;
  long space_mem2;

} dim_str;

#define DOT(A, B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))    // STABLE

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

int isInteger(char *str) {
  if (*str == '\0') {
    return 0;
  }
  for (; *str != '\0'; str++) {
    if (*str < 48 || *str > 57) {
      return 0;
    }
  }
  return 1;
}

template<typename T>
void kernel(T &handle, par_str d_par_gpu, dim_str d_dim_gpu, box_str *d_box_gpu, FOUR_VECTOR *d_rv_gpu, fp *d_qv_gpu,
            FOUR_VECTOR *d_fv_gpu) {
  int bx = handle.get_block(0);
  int tx = handle.get_local(0);
  int wtx = tx;

  if (bx < d_dim_gpu.number_boxes) {

    fp a2 = 2 * d_par_gpu.alpha * d_par_gpu.alpha;

    int first_i;
    [[pacxx::shared]] FOUR_VECTOR rA_shared[100];

    int pointer;
    int k = 0;
    int first_j;
    int j = 0;

    [[pacxx::shared]] FOUR_VECTOR rB_shared[100];
    [[pacxx::shared]] fp qB_shared[100];

    fp r2;
    fp u2;
    fp vij;
    fp fs;
    fp fxij;
    fp fyij;
    fp fzij;
    THREE_VECTOR d;

    first_i = d_box_gpu[bx].offset;

    while (wtx < NUMBER_PAR_PER_BOX) {
      rA_shared[wtx] = d_rv_gpu[first_i + wtx];
      wtx = wtx + NUMBER_THREADS;
    }
    wtx = tx;

    handle.synchronize(); //barrier(CLK_LOCAL_MEM_FENCE);

    for (k = 0; k < (1 + d_box_gpu[bx].nn); k++) {
      if (k == 0) {
        pointer = bx;
      } else {
        pointer = d_box_gpu[bx].nei[k - 1].number;
      }

      first_j = d_box_gpu[pointer].offset;

      while (wtx < NUMBER_PAR_PER_BOX) {
        rB_shared[wtx] = d_rv_gpu[first_j + wtx];
        qB_shared[wtx] = d_qv_gpu[first_j + wtx];
        wtx = wtx + NUMBER_THREADS;
      }
      wtx = tx;

      handle.synchronize();

      while (wtx < NUMBER_PAR_PER_BOX) {
        for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
          r2 = rA_shared[wtx].x + rB_shared[j].x - DOT(rA_shared[wtx], rB_shared[j]);
          u2 = a2 * r2;
          vij = expf(-u2);
          fs = 2 * vij;
          d.x = rA_shared[wtx].x - rB_shared[j].y;
          fxij = fs * d.x;
          d.y = rA_shared[wtx].y - rB_shared[j].z;
          fyij = fs * d.y;
          d.z = rA_shared[wtx].z - rB_shared[j].w;
          fzij = fs * d.z;
          d_fv_gpu[first_i + wtx].x += qB_shared[j] * vij;
          d_fv_gpu[first_i + wtx].y += qB_shared[j] * fxij;
          d_fv_gpu[first_i + wtx].z += qB_shared[j] * fyij;
          d_fv_gpu[first_i + wtx].w += qB_shared[j] * fzij;

        }
        wtx = wtx + NUMBER_THREADS;
      }
      wtx = tx;

      handle.synchronize(); //barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}

void kernel_gpu_opencl_wrapper(par_str par_cpu,
                               dim_str dim_cpu,
                               box_str *box_cpu,
                               FOUR_VECTOR *rv_cpu,
                               fp *qv_cpu,
                               FOUR_VECTOR *fv_cpu) {

  // timer
  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;

  time0 = get_time();

#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
  Executor::Create<NativeRuntime>(0);
#endif

  auto &exec = Executor::get(0);

  size_t local_work_size[1];
  local_work_size[0] = NUMBER_THREADS;
  size_t global_work_size[1];
  global_work_size[0] = dim_cpu.number_boxes * local_work_size[0];

  printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n",
         global_work_size[0] / local_work_size[0],
         local_work_size[0]);

  time1 = get_time();

  auto &d_box_gpu = exec.allocate<box_str>(dim_cpu.number_boxes);
  auto &d_rv_gpu = exec.allocate<FOUR_VECTOR>(dim_cpu.space_elem);
  auto &d_qv_gpu = exec.allocate<fp>(dim_cpu.space_elem);
  auto &d_fv_gpu = exec.allocate<FOUR_VECTOR>(dim_cpu.space_elem);

  time2 = get_time();

  d_box_gpu.upload(box_cpu, dim_cpu.number_boxes);
  d_rv_gpu.upload(rv_cpu, dim_cpu.space_elem);
  d_qv_gpu.upload(qv_cpu, dim_cpu.space_elem);
  d_fv_gpu.upload(fv_cpu, dim_cpu.space_elem);

  auto p_box_gpu = d_box_gpu.get();
  auto p_rv_gpu = d_rv_gpu.get();
  auto p_qv_gpu = d_qv_gpu.get();
  auto p_fv_gpu = d_fv_gpu.get();

  time3 = get_time();

  exec.launch([=](auto &handle) {
    kernel(handle, par_cpu, dim_cpu,
           p_box_gpu,
           p_rv_gpu,
           p_qv_gpu,
           p_fv_gpu);
  }, {{static_cast<size_t>(dim_cpu.number_boxes)}, {NUMBER_THREADS}});

  exec.synchronize();

  time4 = get_time();

  d_fv_gpu.download(fv_cpu, dim_cpu.space_elem);

  time5 = get_time();

  printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

  printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n",
         (float) (time1 - time0) / 1000000,
         (float) (time1 - time0) / (float) (time6 - time0) * 100);
  printf("%15.12f s, %15.12f % : GPU MEM: ALO\n",
         (float) (time2 - time1) / 1000000,
         (float) (time2 - time1) / (float) (time6 - time0) * 100);
  printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",
         (float) (time3 - time2) / 1000000,
         (float) (time3 - time2) / (float) (time6 - time0) * 100);

  printf("%15.12f s, %15.12f % : GPU: KERNEL\n",
         (float) (time4 - time3) / 1000000,
         (float) (time4 - time3) / (float) (time6 - time0) * 100);

  printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",
         (float) (time5 - time4) / 1000000,
         (float) (time5 - time4) / (float) (time6 - time0) * 100);

  printf("Total time:\n");
  printf("%.12f s\n", (float) (time6 - time0) / 1000000);

}

int main(int argc, char *argv[]) {

  // timer
  long long time0;

  time0 = get_time();

  // timer
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;
  long long time7;

  // counters
  int i, j, k, l, m, n;

  // system memory
  par_str par_cpu;
  dim_str dim_cpu;
  box_str *box_cpu;
  FOUR_VECTOR *rv_cpu;
  fp *qv_cpu;
  FOUR_VECTOR *fv_cpu;
  int nh;

  printf("WG size of kernel = %d \n", NUMBER_THREADS);

  time1 = get_time();

  // assing default values
  dim_cpu.arch_arg = 0;
  dim_cpu.cores_arg = 1;
  dim_cpu.boxes1d_arg = 1;

  // go through arguments
  if (argc == 3) {
    for (dim_cpu.cur_arg = 1; dim_cpu.cur_arg < argc; dim_cpu.cur_arg++) {
      // check if -boxes1d
      if (strcmp(argv[dim_cpu.cur_arg], "-boxes1d") == 0) {
        // check if value provided
        if (argc >= dim_cpu.cur_arg + 1) {
          // check if value is a number
          if (isInteger(argv[dim_cpu.cur_arg + 1]) == 1) {
            dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg + 1]);
            if (dim_cpu.boxes1d_arg < 0) {
              printf("ERROR: Wrong value to -boxes1d argument, cannot be <=0\n");
              return 0;
            }
            dim_cpu.cur_arg = dim_cpu.cur_arg + 1;
          }
            // value is not a number
          else {
            printf("ERROR: Value to -boxes1d argument in not a number\n");
            return 0;
          }
        }
          // value not provided
        else {
          printf("ERROR: Missing value to -boxes1d argument\n");
          return 0;
        }
      }
        // unknown
      else {
        printf("ERROR: Unknown argument\n");
        return 0;
      }
    }
    // Print configuration
    printf("Configuration used: arch = %d, cores = %d, boxes1d = %d\n",
           dim_cpu.arch_arg,
           dim_cpu.cores_arg,
           dim_cpu.boxes1d_arg);
  } else {
    printf("Provide boxes1d argument, example: -boxes1d 16");
    return 0;
  }

  time2 = get_time();

  par_cpu.alpha = 0.5;

  time3 = get_time();


  // total number of boxes
  dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg; // 8*8*8=512

  // how many particles space has in each direction
  dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;                            //512*100=51,200
  dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
  dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

  // box array
  dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

  time4 = get_time();

  // allocate boxes
  box_cpu = (box_str *) malloc(dim_cpu.box_mem);

  // initialize number of home boxes
  nh = 0;

  // home boxes in z direction
  for (i = 0; i < dim_cpu.boxes1d_arg; i++) {
    // home boxes in y direction
    for (j = 0; j < dim_cpu.boxes1d_arg; j++) {
      // home boxes in x direction
      for (k = 0; k < dim_cpu.boxes1d_arg; k++) {

        // current home box
        box_cpu[nh].x = k;
        box_cpu[nh].y = j;
        box_cpu[nh].z = i;
        box_cpu[nh].number = nh;
        box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

        // initialize number of neighbor boxes
        box_cpu[nh].nn = 0;

        // neighbor boxes in z direction
        for (l = -1; l < 2; l++) {
          // neighbor boxes in y direction
          for (m = -1; m < 2; m++) {
            // neighbor boxes in x direction
            for (n = -1; n < 2; n++) {

              // check if (this neighbor exists) and (it is not the same as home box)
              if ((((i + l) >= 0 && (j + m) >= 0 && (k + n) >= 0) == true
                  && ((i + l) < dim_cpu.boxes1d_arg && (j + m) < dim_cpu.boxes1d_arg && (k + n) < dim_cpu.boxes1d_arg)
                      == true) &&
                  (l == 0 && m == 0 && n == 0) == false) {

                // current neighbor box
                box_cpu[nh].nei[box_cpu[nh].nn].x = (k + n);
                box_cpu[nh].nei[box_cpu[nh].nn].y = (j + m);
                box_cpu[nh].nei[box_cpu[nh].nn].z = (i + l);
                box_cpu[nh].nei[box_cpu[nh].nn].number =
                    (box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) +
                        (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) +
                        box_cpu[nh].nei[box_cpu[nh].nn].x;
                box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

                // increment neighbor box
                box_cpu[nh].nn = box_cpu[nh].nn + 1;

              }

            } // neighbor boxes in x direction
          } // neighbor boxes in y direction
        } // neighbor boxes in z direction

        // increment home box
        nh = nh + 1;

      } // home boxes in x direction
    } // home boxes in y direction
  } // home boxes in z direction

  //====================================================================================================100
  //	PARAMETERS, DISTANCE, CHARGE AND FORCE
  //====================================================================================================100

  // random generator seed set to random value - time in this case
  srand(42);

  // input (distances)
  rv_cpu = (FOUR_VECTOR *) malloc(dim_cpu.space_mem);
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    rv_cpu[i].x = (rand() % 10 + 1) / 10.0;            // get a number in the range 0.1 - 1.0
    // rv_cpu[i].v = 0.1;			// get a number in the range 0.1 - 1.0
    rv_cpu[i].y = (rand() % 10 + 1) / 10.0;            // get a number in the range 0.1 - 1.0
    // rv_cpu[i].x = 0.2;			// get a number in the range 0.1 - 1.0
    rv_cpu[i].z = (rand() % 10 + 1) / 10.0;            // get a number in the range 0.1 - 1.0
    // rv_cpu[i].y = 0.3;			// get a number in the range 0.1 - 1.0
    rv_cpu[i].w = (rand() % 10 + 1) / 10.0;            // get a number in the range 0.1 - 1.0
    // rv_cpu[i].z = 0.4;			// get a number in the range 0.1 - 1.0
  }

  // input (charge)
  qv_cpu = (fp *) malloc(dim_cpu.space_mem2);
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    qv_cpu[i] = (rand() % 10 + 1) / 10.0;            // get a number in the range 0.1 - 1.0
    // qv_cpu[i] = 0.5;			// get a number in the range 0.1 - 1.0
  }

  // output (forces)
  fv_cpu = (FOUR_VECTOR *) malloc(dim_cpu.space_mem);
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    fv_cpu[i].x = 0;                                // set to 0, because kernels keeps adding to initial value
    fv_cpu[i].y = 0;                                // set to 0, because kernels keeps adding to initial value
    fv_cpu[i].z = 0;                                // set to 0, because kernels keeps adding to initial value
    fv_cpu[i].w = 0;                                // set to 0, because kernels keeps adding to initial value
  }

  time5 = get_time();

  //======================================================================================================================================================150
  //	KERNEL
  //======================================================================================================================================================150

  //====================================================================================================100
  //	GPU_OPENCL
  //====================================================================================================100

  kernel_gpu_opencl_wrapper(par_cpu,
                            dim_cpu,
                            box_cpu,
                            rv_cpu,
                            qv_cpu,
                            fv_cpu);

  time6 = get_time();

  // dump results

  FILE *fptr;
  fptr = fopen("result.txt", "w");
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    fprintf(fptr, "%f, %f, %f, %f\n", fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z, fv_cpu[i].w);
  }
  fclose(fptr);


  free(rv_cpu);
  free(qv_cpu);
  free(fv_cpu);
  free(box_cpu);

  time7 = get_time();

  return 0;

}
