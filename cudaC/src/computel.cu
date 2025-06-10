#include "cuda_value.h"


// 导出到python的函数
extern "C" 
float compute_l(float l, float * trans_tau_d, int T) {

    printf("calling is start\n");

    init_global_XYZEW_V();

    printf("init_global_XYZEW_V is done\n");
    
    float a3 = 1.00/(T/P);

    // 这一段后续优化为宏
    // MIN_XYZ, INITIAL_INVESTMENT, SCALE_TO_INT_X, SCALE_TO_INT_Y, SCALE_TO_INT_Z, SIZE_Z
    int X_index = (int)floorf((INITIAL_INVESTMENT - MIN_XYZ) * SCALE_TO_INT_X);
    int Y_index = (int)floorf((INITIAL_INVESTMENT - MIN_XYZ) * SCALE_TO_INT_Y);
    int Z_index_1 = (int)floorf((a3 * INITIAL_INVESTMENT - MIN_XYZ) * SCALE_TO_INT_Z);
    float delta_z = (a3 * INITIAL_INVESTMENT - MIN_XYZ) * SCALE_TO_INT_Z - Z_index_1;
    int Z_index_2 = (int)fminf(Z_index_1 + 1, SIZE_Z - 1);


    int index1 = IDX_V(X_index, Y_index, Z_index_1, 0);
    int index2 = IDX_V(X_index, Y_index, Z_index_2, 0);

    // 设置随机数生成器
    curandStatePhilox4_32_10_t* d_rng_states;
    int num_threads = SIZE_X * SIZE_Y * SIZE_Z * SIZE_E * SIZE_W;
    cudaMalloc(&d_rng_states,  num_threads*sizeof(*d_rng_states));
    setup<<<(num_threads+1023)/1024,1024>>>(d_rng_states, 101, num_threads);

    printf("setup is done\n");


    // 设置block和grid
    dim3 block(1024);
    dim3 grid((num_threads + block.x - 1) / block.x);

    dim3 block2(1024);
    dim3 grid2((SIZE_X * SIZE_Y * SIZE_Z * SIZE_E + block2.x - 1) / block2.x);

    
    for (int t = T-1; t >= 0; t--) {
        

        float P_tau_t = trans_tau_d[t];
        
        printf("kernel is start %d\n", t);
        // 计算V(t)
        XYZEW_kernel<<<grid, block>>>(0, t, d_rng_states, l, a3, P_tau_t);
        CUDA_CHECK(cudaGetLastError());     // launch
        CUDA_CHECK(cudaDeviceSynchronize()); // runtime
        cudaDeviceSynchronize();

        printf("kernel1 is done %d\n", t);
        
        // 计算W的最大值
        V_tp1_kernel<<<grid2, block2>>>(0, t);
        CUDA_CHECK(cudaGetLastError());     // launch
        CUDA_CHECK(cudaDeviceSynchronize()); // runtime

        printf("kernel2 is done %d\n", t);


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
    
    printf("index1 = %d, index2 = %d\n", index1, index2);
    printf("out1 = %f, out2 = %f\n", out1, out2);
    printf("X_index = %d, Y_index = %d, Z_index_1 = %d, Z_index_2 = %d\n", X_index, Y_index, Z_index_1, Z_index_2);
    printf("1/2---对应的账户值是：%f, %f, %f, %f\n", final_X, final_Y, final_Z_1, final_Z_2);


    cudaFree(d_rng_states);
    clean_global_XYZEW_V();

    return output;
}