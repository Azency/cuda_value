// test_init.cu
#include "cuda_value.h"

float trans_tau_np[10] = {0.95227777, 0.9458399, 0.938519, 0.93016787, 0.92060485, 0.9096251, 0.89702214, 0.88261673, 0.86628806, 0.84799892};

// 导出到python的函数
float compute_l(float l, float * trans_tau_d, int T) {
    float a3 = 1.00/(T/P);

    // 这一段后续优化为宏
    // MIN_XYZ, h_INITIAL_INVESTMENT, SCALE_TO_INT_X, SCALE_TO_INT_Y, SCALE_TO_INT_Z, SIZE_Z
    int X_index = (int)floorf((h_INITIAL_INVESTMENT - MIN_XYZ) * SCALE_TO_INT_X);
    int Y_index = (int)floorf((h_INITIAL_INVESTMENT - MIN_XYZ) * SCALE_TO_INT_Y);
    int Z_index_1 = (int)floorf((a3 * h_INITIAL_INVESTMENT - MIN_XYZ) * SCALE_TO_INT_Z);
    float delta_z = (a3 * h_INITIAL_INVESTMENT - MIN_XYZ) * SCALE_TO_INT_Z - Z_index_1;
    int Z_index_2 = (int)fminf(Z_index_1 + 1, SIZE_Z - 1);


    int index1 = IDX_V(X_index, Y_index, Z_index_1, 0);
    int index2 = IDX_V(X_index, Y_index, Z_index_2, 0);

    // 设置随机数生成器
    curandStatePhilox4_32_10_t* d_rng_states;
    int num_threads = SIZE_X * SIZE_Y * SIZE_Z * SIZE_E * SIZE_W;
    cudaMalloc(&d_rng_states,  num_threads*sizeof(*d_rng_states));
    setup<<<(num_threads+1023)/1024,1024>>>(d_rng_states, 101, num_threads);

    // 设置block和grid
    dim3 block(1024);
    dim3 grid((num_threads + block.x - 1) / block.x);

    dim3 block2(1024);
    dim3 grid2((SIZE_X * SIZE_Y * SIZE_Z * SIZE_E + block2.x - 1) / block2.x);

    for (int t = T-1; t >= 0; t--) {
        float P_tau_t = trans_tau_d[t];
        
        // 计算V(t)
        XYZEW_kernel<<<grid, block>>>(0, t, d_rng_states, l, a3, P_tau_t);
        CUDA_CHECK(cudaGetLastError());     // launch
        CUDA_CHECK(cudaDeviceSynchronize()); // runtime
        cudaDeviceSynchronize();


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
    
    printf("index1 = %d, index2 = %d\n", index1, index2);
    printf("out1 = %f, out2 = %f\n", out1, out2);
    printf("X_index = %d, Y_index = %d, Z_index_1 = %d, Z_index_2 = %d\n", X_index, Y_index, Z_index_1, Z_index_2);
    printf("1/2---对应的账户值是：%f, %f, %f, %f\n", final_X, final_Y, final_Z_1, final_Z_2);


    cudaFree(d_rng_states);

    return output;
}

void run(){
    float l = 0.00f;

    init_global_XYZEW_V();

    printf("init_global_XYZEW_V done\n");

    // 设置随机数生成器
    float output = compute_l(l, trans_tau_np, 10);

    clean_global_XYZEW_V();

}

float compute_l2(float l, float * trans_tau_d, int T) {
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
    dim3 block(1024);
    dim3 grid((num_threads + block.x - 1) / block.x);

    dim3 block2(1024);
    dim3 grid2((h_sXYZE + block2.x - 1) / block2.x);

    for (int t = T-1; t >= 0; t--) {
        float P_tau_t = trans_tau_d[t];
        
        // 计算V(t)
        XYZEW_kernel2<<<grid, block>>>(0, t, d_rng_states, l, a3, P_tau_t);
        CUDA_CHECK(cudaGetLastError());     // launch
        CUDA_CHECK(cudaDeviceSynchronize()); // runtime
        cudaDeviceSynchronize();


        // 计算W的最大值
        V_tp1_kernel2<<<grid2, block2>>>(0, t);
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
    
    printf("index1 = %d, index2 = %d\n", index1, index2);
    printf("out1 = %f, out2 = %f\n", out1, out2);
    printf("X_index = %d, Y_index = %d, Z_index_1 = %d, Z_index_2 = %d\n", X_index, Y_index, Z_index_1, Z_index_2);
    printf("1/2---对应的账户值是：%f, %f, %f, %f\n", final_X, final_Y, final_Z_1, final_Z_2);


    cudaFree(d_rng_states);

    return output;
}


void run2(){
    float l = 0.00f;

    init_global_config(
        0, 100, 41,
        0, 100, 41,
        0, 100, 41,
        0, 1,   2,
        0, 100, 41,
        0.15f, 0.025f, 0.05f, 0.05f, 0.2f, 10000, 1, 100.0f
    );

    init_global_XYZEW_V2();

    float output = compute_l2(l, trans_tau_np, 10);

    clean_global_XYZEW_V();



}



int main() {

    run2();
    return 0;
}

// 测试随机数 ----------------------------      ------------------------------  start
// #include <curand_kernel.h>

// __global__ void mc_kernel(curandStatePhilox4_32_10_t *state,
//                           float *payoff, int steps)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     curandStatePhilox4_32_10_t s = state[tid];

//     float S = 100.f;                       // 例如股票现价
//     float mu = 0.06f, sigma = 0.2f, dt = 1.f/252;
//     // for (int t = 0; t < steps; ++t) {
//     //     float z = curand_normal(&s);       // N(0,1)
//     //     S *= __expf((mu - .5f*sigma*sigma)*dt + sigma*sqrtf(dt)*z);
//     // }
//     payoff[tid] = curand_normal(&s);   // 欧式看涨

//     state[tid] = s;                        // 写回
// }

// __global__ void setup(curandStatePhilox4_32_10_t *state, unsigned long seed, int PATHS)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= PATHS) return;
//     /* sequence=tid, offset=0 → 线程独立子流 */
//     curand_init(seed, tid, 0, &state[tid]);
// }

// int main() {
//     const int PATHS = 1<<20;
//     curandStatePhilox4_32_10_t *d_state;
//     float *d_payoff;
//     cudaMalloc(&d_state,  PATHS*sizeof(*d_state));
//     cudaMalloc(&d_payoff, PATHS*sizeof(float));

//     setup<<<PATHS/256,256>>>(d_state, 101);
//     mc_kernel<<<PATHS/256,256>>>(d_state, d_payoff, /*steps=*/252);

//     float h_payoff[PATHS];
//     cudaMemcpy(h_payoff, d_payoff, PATHS*sizeof(float), cudaMemcpyDeviceToHost);
//     printf("h_payoff = %f\n", h_payoff[14]);

//     // 取均值 …
// }


















// 测试随机数 ----------------------------      ------------------------------  end





















// 测试初始化 ----------------------------      ------------------------------  start







    // // 打印数组内容的辅助函数
    // void print_array(const char* name, float* arr, int size) {
    //     std::cout << name << " = [";
    //     for (int i = 0; i < std::min(size, 5); i++) {
    //         std::cout << std::fixed << std::setprecision(2) << arr[i] << " ";
    //     }
    //     if (size > 5) {
    //         std::cout << "... ";
    //         for (int i = size - 5; i < size; i++) {
    //             std::cout << std::fixed << std::setprecision(2) << arr[i] << " ";
    //         }
    //     }
    //     std::cout << "]" << std::endl;
    // }

    // // 验证数组初始化是否正确
    // bool verify_initialization() {
    //     // 分配主机内存用于验证
    //     float *h_X = new float[SIZE_X];
    //     float *h_Y = new float[SIZE_Y];
    //     float *h_Z = new float[SIZE_Z];
    //     int *h_E = new int[SIZE_E];
    //     float *h_W = new float[SIZE_W];
    //     float *h_V = new float[SIZE_X * SIZE_Y * SIZE_Z * SIZE_E];
    //     float *h_V_tp1 = new float[SIZE_X * SIZE_Y * SIZE_Z * SIZE_E];

    //     // 从设备复制数据到主机
    //     cudaMemcpy(h_X, d_X, SIZE_X * sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_Y, d_Y, SIZE_Y * sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_Z, d_Z, SIZE_Z * sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_E, d_E, SIZE_E * sizeof(int), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_W, d_W, SIZE_W * sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_V, d_V, SIZE_X * SIZE_Y * SIZE_Z * SIZE_E * sizeof(float), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_V_tp1, d_V_tp1, SIZE_X * SIZE_Y * SIZE_Z * SIZE_E * sizeof(float), cudaMemcpyDeviceToHost);

    //     // 验证数据
    //     bool success = true;

    //     // 验证 X 数组
    //     for (int i = 0; i < SIZE_X; i++) {
    //         float expected = MIN_XYZ + (MAX_X - MIN_XYZ) * i / (SIZE_X - 1);
    //         if (fabs(h_X[i] - expected) > 1e-6) {
    //             std::cout << "X array verification failed at index " << i 
    //                     << ": expected " << expected << ", got " << h_X[i] << std::endl;
    //             success = false;
    //             break;
    //         }
    //     }

    //     // 验证 E 数组
    //     for (int i = 0; i < SIZE_E; i++) {
    //         if (h_E[i] != i) {
    //             std::cout << "E array verification failed at index " << i 
    //                     << ": expected " << i << ", got " << h_E[i] << std::endl;
    //             success = false;
    //             break;
    //         }
    //     }

    //     // 验证 V 数组的特定位置
    //     for (int x = 0; x < std::min(SIZE_X, 2); x++) {
    //         for (int y = 0; y < std::min(SIZE_Y, 2); y++) {
    //             for (int z = 0; z < std::min(SIZE_Z, 2); z++) {
    //                 for (int e = 0; e < SIZE_E; e++) {
    //                     float min_ZY = fminf(h_Z[z], h_Y[y]);
    //                     float term = (h_Y[y] <= min_ZY) ? 
    //                                 h_Y[y] : 
    //                                 h_Y[y] - A1 * (h_Y[y] - min_ZY);
    //                     float expected = fmaxf(h_X[x], term);
    //                     float actual = h_V[IDX_V(x, y, z, e)];
                        
    //                     if (fabs(actual - expected) > 1e-6) {
    //                         std::cout << "V array verification failed at (" << x << "," << y << "," 
    //                                 << z << "," << e << "): expected " << expected 
    //                                 << ", got " << actual << std::endl;
    //                         success = false;
    //                         break;
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // 验证 V_tp1 是否和 V 一致
    //     for (int i = 0; i < SIZE_X * SIZE_Y * SIZE_Z * SIZE_E; i++) {
    //         if (fabs(h_V[i] - h_V_tp1[i]) > 1e-6) {
    //             std::cout << "V_tp1 array verification failed at index " << i
    //                     << ": expected " << h_V[i] << ", got " << h_V_tp1[i] << std::endl;
    //             success = false;
    //             break;
    //         }
    //     }

    //     // 打印部分内容
    //     print_array("X", h_X, SIZE_X);
    //     print_array("Y", h_Y, SIZE_Y);
    //     print_array("Z", h_Z, SIZE_Z);
    //     print_array("E", reinterpret_cast<float*>(h_E), SIZE_E);
    //     print_array("W", h_W, SIZE_W);

    //     // 释放主机内存
    //     delete[] h_X;
    //     delete[] h_Y;
    //     delete[] h_Z;
    //     delete[] h_E;
    //     delete[] h_W;
    //     delete[] h_V;
    //     delete[] h_V_tp1;

    //     return success;
    // }

    // int main() {
    //     std::cout << "Testing init_global_XYZEW_V..." << std::endl;
    //     init_global_XYZEW_V();
    //     std::cout << "Testing verify_initialization..." << std::endl;
    //     bool ok = verify_initialization();
    //     if (ok) {
    //         std::cout << "Initialization test PASSED." << std::endl;
    //     } else {
    //         std::cout << "Initialization test FAILED." << std::endl;
    //     }

    //     clean_global_XYZEW_V();
    //     return ok ? 0 : 1;
    // }






