#include "cuda_value.h"
#include <vector>
#include <iostream>



int main() {
    // 初始化设备变量
    int t = 0;
    float a3 = 0.01;
    float l = 0.01;

    init_global_XYZEW_V();

    // 设置随机数生成器
    curandStatePhilox4_32_10_t* d_rng_states;
    int num_threads = SIZE_X * SIZE_Y * SIZE_Z * SIZE_E * SIZE_W;
    init_random_state(d_rng_states, 101, num_threads);
    
    // 初始化随机数生成器
    dim3 block(1024);
    dim3 grid((num_threads + block.x - 1) / block.x);

    XYZEW_kernel<<<grid, block>>>(0, d_V_tp1, d_rng_states, l, t, a3);
    // 需要实现一个初始化随机数生成器的kernel
    // init_rng_kernel<<<grid, block>>>(d_rng_states, time(NULL));

    // 主循环
    // ... 实现主循环逻辑 ...

    // 清理
    clean_global_XYZEW_V();
    cudaFree(d_rng_states);

    return 0;
}