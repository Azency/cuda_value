#ifndef CUDA_VALUE_H
#define CUDA_VALUE_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
// #include "config.h"
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

extern float h_MIN_X, h_MIN_Y, h_MIN_Z, h_MIN_W;
extern float h_MAX_X, h_MAX_Y, h_MAX_Z, h_MAX_W;
extern int h_SIZE_X, h_SIZE_Y, h_SIZE_Z, h_SIZE_E, h_SIZE_W;

extern int h_sWEYZX, h_sEYZX, h_sYZX, h_sZX, h_sX;

extern float h_SCALE_TO_INT_X, h_SCALE_TO_INT_Y, h_SCALE_TO_INT_Z;

extern float h_A1,h_P, h_INITIAL_INVESTMENT, h_DELTA_T;


extern float *d_X, *d_Y, *d_Z, *d_W, *d_V, *d_V_tp1, *d_results;
extern int *d_E;

extern cudaArray_t cuArray0;
extern cudaArray_t cuArray1;
// 描述我们数据的格式（单通道32位浮点数）
extern cudaChannelFormatDesc channelDesc;

extern cudaTextureObject_t texObj0;
extern cudaTextureObject_t texObj1;
extern cudaSurfaceObject_t surfObj0;
extern cudaSurfaceObject_t surfObj1;

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
__global__ void WEYZX_kernel(int offset, int t, curandStatePhilox4_32_10_t *rng_states, float l, float a3, float P_tau_t);

__global__ void test_array_kernel(cudaTextureObject_t texObj0, cudaTextureObject_t texObj1);


__global__ void V_tp1_kernel(int offset, int t);


// 随机数发生器
__global__ void setup(curandStatePhilox4_32_10_t *state, unsigned long seed, int PATHSs);



// 辅助函数声明
__device__ int IDX_V(int e, int y, int z, int x);

__host__ int h_IDX_V(int e, int y, int z, int x);

void init_global_config(
    float min_X, float max_X, int size_X,
    float min_Y, float max_Y, int size_Y,
    float min_Z, float max_Z, int size_Z,
    int min_E, int max_E, int size_E,
    float min_W, float max_W, int size_W,
    float a1, float a2, float r, float mu, float sigma, int motecalo_nums, float p, float initial_investment
);

void init_global_XYZEW_V();

void clean_global_XYZEW_V();

void init_random_state();

void init_texture_surface_object();

void copy_cudaarray_to_vtp1();

void reset_Vtp1();

float compute_l(float l, float * trans_tau_d, int T);

#endif