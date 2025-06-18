#include "cuda_value.h"


// 导出到python的函数
extern "C" 
float pycompute_l(float l, float * trans_tau_d, int T) {
    float a3 = 1.00/(T/h_P);

    // 这一段后续优化为宏
    // MIN_XYZ, h_INITIAL_INVESTMENT, SCALE_TO_INT_X, SCALE_TO_INT_Y, SCALE_TO_INT_Z, SIZE_Z
    int X_index = (int)floorf((h_INITIAL_INVESTMENT - h_MIN_X) * h_SCALE_TO_INT_X);
    int Y_index = (int)floorf((h_INITIAL_INVESTMENT - h_MIN_Y) * h_SCALE_TO_INT_Y);
    int Z_index_1 = (int)floorf((a3 * h_INITIAL_INVESTMENT - h_MIN_Z) * h_SCALE_TO_INT_Z);
    float delta_z = (a3 * h_INITIAL_INVESTMENT - h_MIN_Z) * h_SCALE_TO_INT_Z - Z_index_1;
    int Z_index_2 = (int)fminf(Z_index_1 + 1, h_SIZE_Z - 1);


    int index1 = h_IDX_V(X_index, Y_index, Z_index_1, 0);
    int index2 = h_IDX_V(X_index, Y_index, Z_index_2, 0);

    // 设置随机数生成器
    curandStatePhilox4_32_10_t* d_rng_states;
    int num_threads = h_sXYZEW;
    cudaMalloc(&d_rng_states,  num_threads*sizeof(*d_rng_states));
    setup<<<(num_threads+1023)/1024,1024>>>(d_rng_states, 101, num_threads);

    // 设置block和grid
    dim3 block(512);
    dim3 grid((h_sXYZEW + block.x - 1) / block.x);

    dim3 block2(512);
    dim3 grid2((h_sXYZE + block2.x - 1) / block2.x);
    for (int t = T-1; t >= 0; t--) {
        float P_tau_t = trans_tau_d[t];
        
        // 计算V(t)
        XYZEW_kernel<<<grid, block>>>(0, t, d_rng_states, l, a3, P_tau_t);
        CUDA_CHECK(cudaGetLastError());     // launch
        CUDA_CHECK(cudaDeviceSynchronize()); // runtime

        // 计算W的最大值
        V_tp1_kernel<<<grid2, block2>>>(0, t);
        CUDA_CHECK(cudaGetLastError());     // launch
        CUDA_CHECK(cudaDeviceSynchronize()); // runtime


    }

    float out1, out2;
    cudaMemcpy(&out1, &d_V_tp1[index1], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out2, &d_V_tp1[index2], sizeof(float), cudaMemcpyDeviceToHost);
  
    float output = out1 + (out2 - out1)*delta_z;


    float final_X, final_Y, final_Z_1, final_Z_2;
    cudaMemcpy(&final_X, &d_X[X_index], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_Y, &d_Y[Y_index], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_Z_1, &d_Z[Z_index_1], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_Z_2, &d_Z[Z_index_2], sizeof(float), cudaMemcpyDeviceToHost);
    
    // printf("index1 = %d, index2 = %d\n", index1, index2);
    printf("out1 = %f, out2 = %f, output = %f\n", out1, out2, output);
    printf("X_index = %d, Y_index = %d, Z_index_1 = %d, Z_index_2 = %d\n", X_index, Y_index, Z_index_1, Z_index_2);
    printf("1/2---对应的账户值是：%f, %f, %f, %f\n", final_X, final_Y, final_Z_1, final_Z_2);


    cudaFree(d_rng_states);

    return output;
}



extern "C"
void pyinit_global_XYZEW_V() {
    init_global_XYZEW_V();
}

extern "C"
void pyclean_global_XYZEW_V() {
    clean_global_XYZEW_V();
}

extern "C"
void pyreset_Vtp1() {
    reset_Vtp1();
}

extern "C"
void pyinit_global_config(
    int min_X, int max_X, int size_X,
    int min_Y, int max_Y, int size_Y,
    int min_Z, int max_Z, int size_Z,
    int min_E, int max_E, int size_E,
    int min_W, int max_W, int size_W,
    float a1, float a2, float r, float mu, float sigma, int motecalo_nums, float p, float initial_investment
) {
    init_global_config(
        min_X, max_X, size_X, 
        min_Y, max_Y, size_Y, 
        min_Z, max_Z, size_Z, 
        min_E, max_E, size_E, 
        min_W, max_W, size_W, 
        a1, a2, r, mu, sigma, motecalo_nums, p, initial_investment);
}