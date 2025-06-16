#ifndef CUDA_VALUE_H
#define CUDA_VALUE_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h> 


#define CUDA_CHECK(call)                                \
do {                                                    \
    cudaError_t err = (call);                           \
    if(err != cudaSuccess){                             \
        fprintf(stderr,"CUDA %s:%d %s\n",               \
                __FILE__,__LINE__,cudaGetErrorString(err)); \
        cudaDeviceReset(); exit(EXIT_FAILURE);          \
    }                                                   \
} while(0)

// 定义常量宏 start
#define A1 0.15f
#define A2 0.025f
#define R 0.05f

#define MU 0.05f
#define SIGMA 0.2f
#define MOTECALO_NUMS 10000

#define MIN_XYZ 0
#define MAX_X 100
#define MAX_Y 100
#define MAX_Z 100
#define MAX_W 100


#define SIZE_X 21
#define SIZE_Y 21
#define SIZE_Z 21
#define SIZE_E 2
#define SIZE_W 21
// 定义常量宏 end

#define SCALE_TO_INT_X ((float)(SIZE_X-1) / (MAX_X - MIN_XYZ))
#define SCALE_TO_INT_Y ((float)(SIZE_Y-1) / (MAX_Y - MIN_XYZ))
#define SCALE_TO_INT_Z ((float)(SIZE_Z-1) / (MAX_Z - MIN_XYZ))

#define sXYZEW (SIZE_X*SIZE_Y*SIZE_Z*SIZE_E*SIZE_W)
#define sYZEW (SIZE_Y*SIZE_Z*SIZE_E*SIZE_W)
#define sZEW (SIZE_Z*SIZE_E*SIZE_W)
#define sEW (SIZE_E*SIZE_W)
#define sW (SIZE_W)

#define sXYZE (SIZE_X*SIZE_Y*SIZE_Z*SIZE_E)
#define sYZE (SIZE_Y*SIZE_Z*SIZE_E)
#define sZE (SIZE_Z*SIZE_E)
#define sE (SIZE_E)


#define X0 67
#define X_END 92
#define P 1
#define Q 1
#define INITIAL_INVESTMENT 100.0f
#define DELTA_T 1.0f/P

// 设备函数声明
__device__ float monte_carlo_simulation(
    float XmW,           // X - W
    float Y_tp1,         // Y(t+1)
    float Z_tp1,         // Z(t+1)
    int E_tp1,           // E(t+1)
    float P_tau_tp1,     // P(tau=t+1)
    float P_tau_gep_tp1, // P(tau>=t+1)
    float l,             // 费用率
    curandStatePhilox4_32_10_t * rng_state, // 随机数状态
    int idx              // 线程索引
    );

__device__ float lookup_V(float X, float Y, float Z, int E);

// Kernel 函数声明
__global__ void XYZEW_kernel(int offset, int t, curandStatePhilox4_32_10_t *rng_states, float l, float a3, float P_tau_t);

__global__ void V_tp1_kernel(int offset, int t);

// 随机数发生器
__global__ void setup(curandStatePhilox4_32_10_t *state, unsigned long seed, int PATHSs);

void init_random_state(curandStatePhilox4_32_10_t *state, unsigned long seed, int PATHS);



// 辅助函数声明
extern float *d_X, *d_Y, *d_Z, *d_W, *d_V, *d_V_tp1, *d_results;
extern int *d_E;

__constant__ float *d_d_X, *d_d_Y, *d_d_Z, *d_d_W, *d_d_V, *d_d_V_tp1, *d_d_results;
__constant__ int *d_d_E;


__host__ __device__ int IDX_V(int x, int y, int z, int e);

void init_global_XYZEW_V();

void clean_global_XYZEW_V();

void reset_Vtp1();

#endif