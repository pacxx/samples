/********************************************************************
	euler3d.cpp
	: parallelized code of CFD

	- original code from the AIAA-2009-4001 by Andrew Corrigan, acorriga@gmu.edu
	- parallelization with OpenCL API has been applied by
	Jianbin Fang - j.fang@tudelft.nl
	Delft University of Technology
	Faculty of Electrical Engineering, Mathematics and Computer Science
	Department of Software Technology
	Parallel and Distributed Systems Group
	on 24/03/2011
********************************************************************/

#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <PACXX.h>
#include "util.h"
using namespace pacxx::v2;
/*
 * Options
 *
 */

#define iterations 1
#ifndef block_length
#define block_length 192
#endif

#define NDIM 3
#define NNB 4

#define RK 3    // 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

/*
 * not options
 */


#if block_length > 128
#warning "the kernels may fail too launch on some systems if the block length is too large"
#endif

#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)

/* ============================================================
//--functions: 	kernel funtion
//--programmer:	Jianbin Fang
//--date:		24/03/2011
============================================================ */

#define GAMMA (1.4f)

#define NDIM 3
#define NNB 4

#define RK 3    // 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)
//#pragma OPENCL EXTENSION CL_MAD : enable

//self-defined user type
typedef struct {
  float x;
  float y;
  float z;
} myfloat3;
/*------------------------------------------------------------
	@function:	set memory
	@params:
		mem_d: 		target memory to be set;
		val:		set the target memory to value 'val'
		num_bytes:	the number of bytes all together
	@return:	through mem_d
------------------------------------------------------------*/
template<typename CFG>
void memset_kernel(char *mem_d, short val, int ct, CFG &cfg) {
  const int thread_id = cfg.get_global(0);
  if (thread_id >= ct)
    return;
  mem_d[thread_id] = val;
}

//--cambine: omit &
inline void compute_velocity(float density, myfloat3 momentum, myfloat3 *velocity) {
  velocity->x = momentum.x / density;
  velocity->y = momentum.y / density;
  velocity->z = momentum.z / density;
}

inline float compute_speed_sqd(myfloat3 velocity) {
  return velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z;
}

inline float compute_pressure(float density, float density_energy, float speed_sqd) {
  return ((float) (GAMMA) - (float) (1.0f)) * (density_energy - (float) (0.5f) * density * speed_sqd);
}
inline float compute_speed_of_sound(float density, float pressure) {
  //return sqrtf(float(GAMMA)*pressure/density);
  return sqrtf((float) (GAMMA) * pressure / density);
}
inline void compute_flux_contribution(float density,
                                      myfloat3 momentum,
                                      float density_energy,
                                      float pressure,
                                      myfloat3 velocity,
                                      myfloat3 *fc_momentum_x,
                                      myfloat3 *fc_momentum_y,
                                      myfloat3 *fc_momentum_z,
                                      myfloat3 *fc_density_energy) {
  fc_momentum_x->x = velocity.x * momentum.x + pressure;
  fc_momentum_x->y = velocity.x * momentum.y;
  fc_momentum_x->z = velocity.x * momentum.z;

  fc_momentum_y->x = fc_momentum_x->y;
  fc_momentum_y->y = velocity.y * momentum.y + pressure;
  fc_momentum_y->z = velocity.y * momentum.z;

  fc_momentum_z->x = fc_momentum_x->z;
  fc_momentum_z->y = fc_momentum_y->z;
  fc_momentum_z->z = velocity.z * momentum.z + pressure;

  float de_p = density_energy + pressure;
  fc_density_energy->x = velocity.x * de_p;
  fc_density_energy->y = velocity.y * de_p;
  fc_density_energy->z = velocity.z * de_p;
}

template<typename CFG>
void initialize_variables(float *variables, float *ff_variable, int nelr, CFG &cfg) {
  const int i = cfg.get_global(0);
  if (i >= nelr)
    return;
  for (int j = 0; j < NVAR; j++)
    variables[i + j * nelr] = ff_variable[j];

}

template<typename CFG>
void compute_step_factor(float *variables,
                         float *areas,
                         float *step_factors,
                         int nelr, CFG &cfg) {
  const int i = cfg.get_global(0);
  if (i >= nelr)
    return;

  float density = variables[i + VAR_DENSITY * nelr];
  myfloat3 momentum;
  momentum.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
  momentum.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
  momentum.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

  float density_energy = variables[i + VAR_DENSITY_ENERGY * nelr];

  myfloat3 velocity;
  compute_velocity(density, momentum, &velocity);
  float speed_sqd = compute_speed_sqd(velocity);
  //float speed_sqd;
  //compute_speed_sqd(velocity, speed_sqd);
  float pressure = compute_pressure(density, density_energy, speed_sqd);
  float speed_of_sound = compute_speed_of_sound(density, pressure);

  // dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
  //step_factors[i] = (float)(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
  step_factors[i] = (float) (0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
}

template<typename CFG>
void compute_flux(
    int *elements_surrounding_elements,
    float *normals,
    float *variables,
    float *ff_variable,
    float *fluxes,
    myfloat3 *ff_flux_contribution_density_energy,
    myfloat3 *ff_flux_contribution_momentum_x,
    myfloat3 *ff_flux_contribution_momentum_y,
    myfloat3 *ff_flux_contribution_momentum_z,
    int nelr, CFG &cfg) {
  const float smoothing_coefficient = (float) (0.2f);
//const int i = (blockDim.x*blockIdx.x + threadIdx.x);
  const int i = cfg.get_global(0);
  if (i >= nelr)
    return;
  int j, nb;
  myfloat3 normal;
  float normal_len;
  float factor;

  float density_i = variables[i + VAR_DENSITY * nelr];
  myfloat3 momentum_i;
  momentum_i.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
  momentum_i.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
  momentum_i.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

  float density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];

  myfloat3 velocity_i;
  compute_velocity(density_i, momentum_i, &velocity_i);
  float speed_sqd_i = compute_speed_sqd(velocity_i);
//float speed_sqd_i;
//compute_speed_sqd(velocity_i, speed_sqd_i);
//float speed_i                              = sqrtf(speed_sqd_i);
  float speed_i = sqrtf(speed_sqd_i);
  float pressure_i = compute_pressure(density_i, density_energy_i, speed_sqd_i);
  float speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
  myfloat3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
  myfloat3 flux_contribution_i_density_energy;
  compute_flux_contribution(density_i,
                            momentum_i,
                            density_energy_i,
                            pressure_i,
                            velocity_i,
                            &flux_contribution_i_momentum_x,
                            &flux_contribution_i_momentum_y,
                            &flux_contribution_i_momentum_z,
                            &flux_contribution_i_density_energy);

  float flux_i_density = (float) (0.0f);
  myfloat3 flux_i_momentum;
  flux_i_momentum.x = (float) (0.0f);
  flux_i_momentum.y = (float) (0.0f);
  flux_i_momentum.z = (float) (0.0f);
  float flux_i_density_energy = (float) (0.0f);

  myfloat3 velocity_nb;
  float density_nb, density_energy_nb;
  myfloat3 momentum_nb;
  myfloat3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
  myfloat3 flux_contribution_nb_density_energy;
  float speed_sqd_nb, speed_of_sound_nb, pressure_nb;

#pragma unroll
  for (j = 0; j < NNB; j++) {
    nb = elements_surrounding_elements[i + j * nelr];
    normal.x = normals[i + (j + 0 * NNB) * nelr];
    normal.y = normals[i + (j + 1 * NNB) * nelr];
    normal.z = normals[i + (j + 2 * NNB) * nelr];
//normal_len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
    normal_len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

    if (nb >= 0)    // a legitimate neighbor
    {
      density_nb = variables[nb + VAR_DENSITY * nelr];
      momentum_nb.x = variables[nb + (VAR_MOMENTUM + 0) * nelr];
      momentum_nb.y = variables[nb + (VAR_MOMENTUM + 1) * nelr];
      momentum_nb.z = variables[nb + (VAR_MOMENTUM + 2) * nelr];
      density_energy_nb = variables[nb + VAR_DENSITY_ENERGY * nelr];
      compute_velocity(density_nb, momentum_nb, &velocity_nb);
      speed_sqd_nb = compute_speed_sqd(velocity_nb);
      pressure_nb = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
      speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);
      compute_flux_contribution(density_nb,
                                momentum_nb,
                                density_energy_nb,
                                pressure_nb,
                                velocity_nb,
                                &flux_contribution_nb_momentum_x,
                                &flux_contribution_nb_momentum_y,
                                &flux_contribution_nb_momentum_z,
                                &flux_contribution_nb_density_energy);

// artificial viscosity
      factor = -normal_len * smoothing_coefficient * (float) (0.5f)
          * (speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
      flux_i_density += factor * (density_i - density_nb);
      flux_i_density_energy += factor * (density_energy_i - density_energy_nb);
      flux_i_momentum.x += factor * (momentum_i.x - momentum_nb.x);
      flux_i_momentum.y += factor * (momentum_i.y - momentum_nb.y);
      flux_i_momentum.z += factor * (momentum_i.z - momentum_nb.z);

// accumulate cell-centered fluxes
      factor = (float) (0.5f) * normal.x;
      flux_i_density += factor * (momentum_nb.x + momentum_i.x);
      flux_i_density_energy += factor * (flux_contribution_nb_density_energy.x + flux_contribution_i_density_energy.x);
      flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.x + flux_contribution_i_momentum_x.x);
      flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.x + flux_contribution_i_momentum_y.x);
      flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.x + flux_contribution_i_momentum_z.x);

      factor = (float) (0.5f) * normal.y;
      flux_i_density += factor * (momentum_nb.y + momentum_i.y);
      flux_i_density_energy += factor * (flux_contribution_nb_density_energy.y + flux_contribution_i_density_energy.y);
      flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.y + flux_contribution_i_momentum_x.y);
      flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.y + flux_contribution_i_momentum_y.y);
      flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.y + flux_contribution_i_momentum_z.y);

      factor = (float) (0.5f) * normal.z;
      flux_i_density += factor * (momentum_nb.z + momentum_i.z);
      flux_i_density_energy += factor * (flux_contribution_nb_density_energy.z + flux_contribution_i_density_energy.z);
      flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.z + flux_contribution_i_momentum_x.z);
      flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.z + flux_contribution_i_momentum_y.z);
      flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.z + flux_contribution_i_momentum_z.z);
    } else if (nb == -1)    // a wing boundary
    {
      flux_i_momentum.x += normal.x * pressure_i;
      flux_i_momentum.y += normal.y * pressure_i;
      flux_i_momentum.z += normal.z * pressure_i;
    } else if (nb == -2) // a far field boundary
    {
      factor = (float) (0.5f) * normal.x;
      flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 0] + momentum_i.x);
      flux_i_density_energy +=
          factor * (ff_flux_contribution_density_energy[0].x + flux_contribution_i_density_energy.x);
      flux_i_momentum.x += factor * (ff_flux_contribution_momentum_x[0].x + flux_contribution_i_momentum_x.x);
      flux_i_momentum.y += factor * (ff_flux_contribution_momentum_y[0].x + flux_contribution_i_momentum_y.x);
      flux_i_momentum.z += factor * (ff_flux_contribution_momentum_z[0].x + flux_contribution_i_momentum_z.x);

      factor = (float) (0.5f) * normal.y;
      flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y);
      flux_i_density_energy +=
          factor * (ff_flux_contribution_density_energy[0].y + flux_contribution_i_density_energy.y);
      flux_i_momentum.x += factor * (ff_flux_contribution_momentum_x[0].y + flux_contribution_i_momentum_x.y);
      flux_i_momentum.y += factor * (ff_flux_contribution_momentum_y[0].y + flux_contribution_i_momentum_y.y);
      flux_i_momentum.z += factor * (ff_flux_contribution_momentum_z[0].y + flux_contribution_i_momentum_z.y);

      factor = (float) (0.5f) * normal.z;
      flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z);
      flux_i_density_energy +=
          factor * (ff_flux_contribution_density_energy[0].z + flux_contribution_i_density_energy.z);
      flux_i_momentum.x += factor * (ff_flux_contribution_momentum_x[0].z + flux_contribution_i_momentum_x.z);
      flux_i_momentum.y += factor * (ff_flux_contribution_momentum_y[0].z + flux_contribution_i_momentum_y.z);
      flux_i_momentum.z += factor * (ff_flux_contribution_momentum_z[0].z + flux_contribution_i_momentum_z.z);

    }
  }

  fluxes[i + VAR_DENSITY * nelr] = flux_i_density;
  fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x;
  fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y;
  fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z;
  fluxes[i + VAR_DENSITY_ENERGY * nelr] = flux_i_density_energy;
}

template<typename CFG>
void time_step(int j, int nelr,
               float *old_variables,
               float *variables,
               float *step_factors,
               float *fluxes, CFG &cfg) {
  //const int i = (blockDim.x*blockIdx.x + threadIdx.x);
  const int i = cfg.get_global(0);
  if (i >= nelr)
    return;

  float factor = step_factors[i] / (float) (RK + 1 - j);

  variables[i + VAR_DENSITY * nelr] = old_variables[i + VAR_DENSITY * nelr] + factor * fluxes[i + VAR_DENSITY * nelr];
  variables[i + VAR_DENSITY_ENERGY * nelr] =
      old_variables[i + VAR_DENSITY_ENERGY * nelr] + factor * fluxes[i + VAR_DENSITY_ENERGY * nelr];
  variables[i + (VAR_MOMENTUM + 0) * nelr] =
      old_variables[i + (VAR_MOMENTUM + 0) * nelr] + factor * fluxes[i + (VAR_MOMENTUM + 0) * nelr];
  variables[i + (VAR_MOMENTUM + 1) * nelr] =
      old_variables[i + (VAR_MOMENTUM + 1) * nelr] + factor * fluxes[i + (VAR_MOMENTUM + 1) * nelr];
  variables[i + (VAR_MOMENTUM + 2) * nelr] =
      old_variables[i + (VAR_MOMENTUM + 2) * nelr] + factor * fluxes[i + (VAR_MOMENTUM + 2) * nelr];

}

template<typename T>
void dump(T &variables, int nel, int nelr) {
  float *h_variables = new float[nelr * NVAR];
  variables.download(h_variables, nelr * NVAR);

  {
    std::ofstream file("density");
    file << nel << " " << nelr << std::endl;
    for (int i = 0; i < nel; i++)
      file << h_variables[i + VAR_DENSITY * nelr] << std::endl;
  }

  {
    std::ofstream file("momentum");
    file << nel << " " << nelr << std::endl;
    for (int i = 0; i < nel; i++) {
      for (int j = 0; j != NDIM; j++)
        file << h_variables[i + (VAR_MOMENTUM + j) * nelr] << " ";
      file << std::endl;
    }
  }

  {
    std::ofstream file("density_energy");
    file << nel << " " << nelr << std::endl;
    for (int i = 0; i < nel; i++)
      file << h_variables[i + VAR_DENSITY_ENERGY * nelr] << std::endl;
  }
  delete[] h_variables;
}

template<typename EXEC, typename T>
void initialize_variables(EXEC &exec, int nelr, T &variables, T &ff_variable) {

  size_t work_items = nelr;
  size_t work_group_size = BLOCK_SIZE_1;

  auto d1 = variables.get();
  auto d2 = ff_variable.get();

  exec.launch([=](auto &cfg) {
    initialize_variables(d1, d2, nelr, cfg);
  }, {{work_items / work_group_size}, {work_group_size}});
}

template<typename EXEC, typename T>
void compute_step_factor(EXEC &exec, int nelr, T &variables, T &areas, T &step_factors) {

  size_t work_items = nelr;
  size_t work_group_size = BLOCK_SIZE_2;

  auto d1 = variables.get();
  auto d2 = areas.get();
  auto d3 = step_factors.get();

  exec.launch([=](auto &cfg) {
    compute_step_factor(d1, d2, d3, nelr, cfg);
  }, {{work_items / work_group_size}, {work_group_size}});
}

template<typename EXEC, typename T1, typename T2, typename T3>
void compute_flux(EXEC &exec, int nelr, T1 &elements_surrounding_elements, T2 &normals, T2 &variables, T2 &ff_variable, \
            T2 &fluxes, T3 &ff_flux_contribution_density_energy,
                  T3 &ff_flux_contribution_momentum_x,
                  T3 &ff_flux_contribution_momentum_y,
                  T3 &ff_flux_contribution_momentum_z) {

  size_t work_items = nelr;
  size_t work_group_size = BLOCK_SIZE_3;

  auto d1 = elements_surrounding_elements.get();
  auto d2 = normals.get();
  auto d3 = variables.get();
  auto d4 = ff_variable.get();
  auto d5 = fluxes.get();
  auto d6 = ff_flux_contribution_density_energy.get();
  auto d7 = ff_flux_contribution_momentum_x.get();
  auto d8 = ff_flux_contribution_momentum_y.get();
  auto d9 = ff_flux_contribution_momentum_z.get();
  exec.launch([=](auto &cfg) {
    compute_flux(d1, d2, d3, d4, d5, d6, d7, d8, d9, nelr, cfg);
  }, {{work_items / work_group_size}, {work_group_size}});
}

template<typename EXEC, typename T>
void time_step(EXEC &exec, int j, int nelr, T &old_variables, T &variables, T &step_factors, T &fluxes) {

  size_t work_items = nelr;
  size_t work_group_size = BLOCK_SIZE_4;

  auto d1 = old_variables.get();
  auto d2 = variables.get();
  auto d3 = step_factors.get();
  auto d4 = fluxes.get();

  exec.launch([=](auto &cfg) {
    time_step(j, nelr, d1, d2, d3, d4, cfg);
  }, {{work_items / work_group_size}, {work_group_size}});
}
inline void compute_flux_contribution(float &density,
                                      myfloat3 &momentum,
                                      float &density_energy,
                                      float &pressure,
                                      myfloat3 &velocity,
                                      myfloat3 &fc_momentum_x,
                                      myfloat3 &fc_momentum_y,
                                      myfloat3 &fc_momentum_z,
                                      myfloat3 &fc_density_energy) {
  fc_momentum_x.x = velocity.x * momentum.x + pressure;
  fc_momentum_x.y = velocity.x * momentum.y;
  fc_momentum_x.z = velocity.x * momentum.z;

  fc_momentum_y.x = fc_momentum_x.y;
  fc_momentum_y.y = velocity.y * momentum.y + pressure;
  fc_momentum_y.z = velocity.y * momentum.z;

  fc_momentum_z.x = fc_momentum_x.z;
  fc_momentum_z.y = fc_momentum_y.z;
  fc_momentum_z.z = velocity.z * momentum.z + pressure;

  float de_p = density_energy + pressure;
  fc_density_energy.x = velocity.x * de_p;
  fc_density_energy.y = velocity.y * de_p;
  fc_density_energy.z = velocity.z * de_p;
}

template<typename EXEC, typename T>
void memset_kernel(EXEC &exec, T &mem_d, short val, int number_bytes) {
  size_t work_group_size = BLOCK_SIZE_0;

  auto memd = mem_d.get();
  exec.launch([=](auto &cfg) {
    memset_kernel(reinterpret_cast<char *>(memd), val, number_bytes, cfg);
  }, {{static_cast<size_t>(number_bytes) / work_group_size}, {work_group_size}});
}

/*
 * Main function
 */
int main(int argc, char **argv) {
  printf(
      "WG size of kernel:initialize = %d, WG size of kernel:compute_step_factor = %d, WG size of kernel:compute_flux = %d, WG size of kernel:time_step = %d\n",
      BLOCK_SIZE_1,
      BLOCK_SIZE_2,
      BLOCK_SIZE_3,
      BLOCK_SIZE_4);

#ifdef USE_EXPERIMENTAL_BACKEND
  // craete the default executor
Executor::Create<NativeRuntime>(0);
#endif

  auto &exec = Executor::get(0);

  if (argc < 2) {
    std::cout << "specify data file name" << std::endl;
    return 0;
  }
  const char *data_file_name = argv[1];

  float h_ff_variable[NVAR];

  // set far field conditions and load them into constant memory on the gpu

  //float h_ff_variable[NVAR];
  const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

  h_ff_variable[VAR_DENSITY] = float(1.4);

  float ff_pressure = float(1.0f);
  float ff_speed_of_sound = sqrt(GAMMA * ff_pressure / h_ff_variable[VAR_DENSITY]);
  float ff_speed = float(ff_mach) * ff_speed_of_sound;

  myfloat3 ff_velocity;
  ff_velocity.x = ff_speed * float(cos((float) angle_of_attack));
  ff_velocity.y = ff_speed * float(sin((float) angle_of_attack));
  ff_velocity.z = 0.0f;

  h_ff_variable[VAR_MOMENTUM + 0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
  h_ff_variable[VAR_MOMENTUM + 1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
  h_ff_variable[VAR_MOMENTUM + 2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;

  h_ff_variable[VAR_DENSITY_ENERGY] =
      h_ff_variable[VAR_DENSITY] * (float(0.5f) * (ff_speed * ff_speed)) + (ff_pressure / float(GAMMA - 1.0f));

  myfloat3 h_ff_momentum;
  h_ff_momentum.x = *(h_ff_variable + VAR_MOMENTUM + 0);
  h_ff_momentum.y = *(h_ff_variable + VAR_MOMENTUM + 1);
  h_ff_momentum.z = *(h_ff_variable + VAR_MOMENTUM + 2);
  myfloat3 h_ff_flux_contribution_momentum_x;
  myfloat3 h_ff_flux_contribution_momentum_y;
  myfloat3 h_ff_flux_contribution_momentum_z;
  myfloat3 h_ff_flux_contribution_density_energy;
  compute_flux_contribution(h_ff_variable[VAR_DENSITY],
                            h_ff_momentum,
                            h_ff_variable[VAR_DENSITY_ENERGY],
                            ff_pressure,
                            ff_velocity,
                            h_ff_flux_contribution_momentum_x,
                            h_ff_flux_contribution_momentum_y,
                            h_ff_flux_contribution_momentum_z,
                            h_ff_flux_contribution_density_energy);

  // copy far field conditions to the gpu
  //cl_mem ff_variable, ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y,ff_flux_contribution_momentum_z,  ff_flux_contribution_density_energy;
  auto &ff_variable = exec.allocate<float>(NVAR);
  auto &ff_flux_contribution_momentum_x = exec.allocate<myfloat3>(1);
  auto &ff_flux_contribution_momentum_y = exec.allocate<myfloat3>(1);
  auto &ff_flux_contribution_momentum_z = exec.allocate<myfloat3>(1);
  auto &ff_flux_contribution_density_energy = exec.allocate<myfloat3>(1);
  ff_variable.upload(h_ff_variable, NVAR);
  ff_flux_contribution_momentum_x.upload(&h_ff_flux_contribution_momentum_x, 1);
  ff_flux_contribution_momentum_y.upload(&h_ff_flux_contribution_momentum_y, 1);
  ff_flux_contribution_momentum_z.upload(&h_ff_flux_contribution_momentum_z, 1);
  ff_flux_contribution_density_energy.upload(&h_ff_flux_contribution_density_energy, 1);

  int nel;
  int nelr;
  // read in domain geometry
  //float* areas;
  //int* elements_surrounding_elements;
  //float* normals;

  std::ifstream file(data_file_name);

  file >> nel;
  nelr = block_length * ((nel / block_length) + std::min(1, nel % block_length));
  std::cout << "--cambine: nel=" << nel << ", nelr=" << nelr << std::endl;
  float *h_areas = new float[nelr];
  int *h_elements_surrounding_elements = new int[nelr * NNB];
  float *h_normals = new float[nelr * NDIM * NNB];


  // read in data
  for (int i = 0; i < nel; i++) {
    file >> h_areas[i];
    for (int j = 0; j < NNB; j++) {
      file >> h_elements_surrounding_elements[i + j * nelr];
      if (h_elements_surrounding_elements[i + j * nelr] < 0)
        h_elements_surrounding_elements[i + j * nelr] = -1;
      h_elements_surrounding_elements[i + j * nelr]--; //it's coming in with Fortran numbering

      for (int k = 0; k < NDIM; k++) {
        file >> h_normals[i + (j + k * NNB) * nelr];
        h_normals[i + (j + k * NNB) * nelr] = -h_normals[i + (j + k * NNB) * nelr];
      }
    }
  }

  // fill in remaining data
  int last = nel - 1;
  for (int i = nel; i < nelr; i++) {
    h_areas[i] = h_areas[last];
    for (int j = 0; j < NNB; j++) {
      // duplicate the last element
      h_elements_surrounding_elements[i + j * nelr] = h_elements_surrounding_elements[last + j * nelr];
      for (int k = 0; k < NDIM; k++)
        h_normals[last + (j + k * NNB) * nelr] = h_normals[last + (j + k * NNB) * nelr];
    }
  }

  auto &areas = exec.allocate<float>(nelr);
  areas.upload(h_areas, nelr);

  auto &elements_surrounding_elements = exec.allocate<int>(nelr * NNB);
  elements_surrounding_elements.upload(h_elements_surrounding_elements, nelr * NNB);

  auto &normals = exec.allocate<float>(nelr * NDIM * NNB);
  normals.upload(h_normals, nelr * NDIM * NNB);

  delete[] h_areas;
  delete[] h_elements_surrounding_elements;
  delete[] h_normals;


  // Create arrays and set initial conditions
  auto &variables = exec.allocate<float>(nelr * NVAR);
  int tp = 0;
  initialize_variables(exec, nelr, variables, ff_variable);
  auto &old_variables = exec.allocate<float>(nelr * NVAR);
  auto &fluxes = exec.allocate<float>(nelr * NVAR);
  auto &step_factors = exec.allocate<float>(nelr);
  // make sure all memory is floatly allocated before we start timing
  initialize_variables(exec, nelr, old_variables, ff_variable);
  initialize_variables(exec, nelr, fluxes, ff_variable);
  memset_kernel(exec, step_factors, 0, sizeof(float) * nelr);
  // make sure CUDA isn't still doing something before we start timing

  // these need to be computed the first time in order to compute time step
  std::cout << "Starting..." << std::endl;

  // Begin iterations
  for (int i = 0; i < iterations; i++) {
    variables.copyTo(old_variables.get());
    // the first iteration we compute the time step
    compute_step_factor(exec, nelr, variables, areas, step_factors);
    for (int j = 0; j < RK; j++) {
      compute_flux(exec, nelr,
                   elements_surrounding_elements,
                   normals,
                   variables,
                   ff_variable,
                   fluxes,
                   ff_flux_contribution_density_energy,
                   ff_flux_contribution_momentum_x,
                   ff_flux_contribution_momentum_y,
                   ff_flux_contribution_momentum_z);

      time_step(exec, j, nelr, old_variables, variables, step_factors, fluxes);
    }
  }
  std::cout << "Saving solution..." << std::endl;
  dump(variables, nel, nelr);
  std::cout << "Saved solution..." << std::endl;

  return 0;
}
