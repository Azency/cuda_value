// test_init.cu
#include "cuda_value.h"
#include <nvtx3/nvToolsExt.h>
//male
// float trans_tau_np[25] = {0.98919799, 0.98850676, 0.98755648, 0.98662116, 0.98551021, 0.98429417, 0.98286998, 0.98122165, 0.97926787, 0.97695839, 0.97422256, 0.97100599, 0.96725252, 0.96291495, 0.95794281, 0.95227777, 0.9458399,  0.938519, 0.93016787, 0.92060485, 0.9096251,  0.89702214, 0.88261673, 0.86628806, 0.84799892}
// float trans_tau_np[10] = {0.95227777, 0.9458399,  0.938519, 0.93016787, 0.92060485, 0.9096251,  0.89702214, 0.88261673, 0.86628806, 0.84799892};

//female
float trans_tau_np[25] = {0.99320371, 0.99275112, 0.99205837, 0.99139464, 0.99058581, 0.98970493, 0.98867356, 0.98749117, 0.98610283, 0.98447572, 0.98255486, 0.98028799, 0.97760967, 0.9744482,  0.97071873, 0.96632442, 0.96115395, 0.95508318, 0.9479776,  0.93969821, 0.93010926, 0.91908749, 0.90653058, 0.89236242, 0.87653274};
// float trans_tau_np[8] = {0.95508318, 0.9479776, 0.93969821, 0.93010926, 0.91908749, 0.90653058, 0.89236242, 0.87653274};

float compute_l(float l, float * trans_tau_d, int T) {
    float a3 = 1.00/(T/h_P);

    // 这一段后续优化为宏
    // MIN_XYZ, h_INITIAL_INVESTMENT, SCALE_TO_INT_X, SCALE_TO_INT_Y, SCALE_TO_INT_Z, SIZE_Z
    int X_index_1= (int)floorf((h_INITIAL_INVESTMENT - h_MIN_X) * h_SCALE_TO_INT_X);
    int X_index_2=(int)fminf(X_index_1 + 1, h_SIZE_X - 1);
    float delta_x = (h_INITIAL_INVESTMENT - h_MIN_X) * h_SCALE_TO_INT_X - X_index_1;

    int Y_index_1 = (int)floorf((h_INITIAL_INVESTMENT - h_MIN_Y) * h_SCALE_TO_INT_Y);
    int Y_index_2 = (int)fminf(Y_index_1 + 1, h_SIZE_Y - 1);
    float delta_y = (h_INITIAL_INVESTMENT - h_MIN_Y) * h_SCALE_TO_INT_Y - Y_index_1;

    int Z_index_1 = (int)floorf((a3 * h_INITIAL_INVESTMENT - h_MIN_Z) * h_SCALE_TO_INT_Z);
    float delta_z = (a3 * h_INITIAL_INVESTMENT - h_MIN_Z) * h_SCALE_TO_INT_Z - Z_index_1;
    int Z_index_2 = (int)fminf(Z_index_1 + 1, h_SIZE_Z - 1);

    int index1 = h_IDX_V(0, Y_index_1, Z_index_1, X_index_1);
    int index2 = h_IDX_V(0, Y_index_1, Z_index_1, X_index_2);   
    int index3 = h_IDX_V(0, Y_index_2, Z_index_1, X_index_1);
    int index4 = h_IDX_V(0, Y_index_1, Z_index_2, X_index_1);
    int index5 = h_IDX_V(0, Y_index_2, Z_index_1, X_index_2);
    int index6 = h_IDX_V(0, Y_index_1, Z_index_2, X_index_2);
    int index7 = h_IDX_V(0, Y_index_2, Z_index_2, X_index_1);
    int index8 = h_IDX_V(0, Y_index_2, Z_index_2, X_index_2);


    // 设置随机数生成器
    curandStatePhilox4_32_10_t* d_rng_states;
    int num_threads = h_sWEYZX;
    cudaMalloc(&d_rng_states,  num_threads*sizeof(*d_rng_states));
    setup<<<(num_threads+1023)/1024,1024>>>(d_rng_states, 101, num_threads);

    // 设置block和grid
    dim3 block(512);
    dim3 grid((h_sWEYZX + block.x - 1) / block.x);

    dim3 block2(512);
    dim3 grid2((h_sEYZX + block2.x - 1) / block2.x);
    for (int t = T-1; t >= 0; t--) {
        float P_tau_t = trans_tau_d[t];
        
        // 计算V(t)
        // t = -1;
        nvtxRangePushA("XYZEW_kernel");
        WEYZX_kernel<<<grid, block>>>(0, t, d_rng_states, l, a3, P_tau_t);
        nvtxRangePop();
        CUDA_CHECK(cudaGetLastError());     // launch
        CUDA_CHECK(cudaDeviceSynchronize()); // runtime

        // 计算W的最大值
        nvtxRangePushA("V_tp1_kernel");
        V_tp1_kernel<<<grid2, block2>>>(0, t);
        nvtxRangePop();
        CUDA_CHECK(cudaGetLastError());     // launch
        CUDA_CHECK(cudaDeviceSynchronize()); // runtime


    }

    copy_cudaarray_to_vtp1();

    float out1, out2, out3, out4, out5, out6, out7, out8;
    cudaMemcpy(&out1, &d_V_tp1[index1], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out2, &d_V_tp1[index2], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out3, &d_V_tp1[index3], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out4, &d_V_tp1[index4], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out5, &d_V_tp1[index5], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out6, &d_V_tp1[index6], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out7, &d_V_tp1[index7], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out8, &d_V_tp1[index8], sizeof(float), cudaMemcpyDeviceToHost);


    float output = (1-delta_x) * (1-delta_y) * (1-delta_z) * out1
                 + delta_x * (1-delta_y) * (1-delta_z) * out2
                 + (1-delta_x) * delta_y * (1-delta_z) * out3
                 + (1-delta_x) * (1-delta_y) * delta_z * out4
                 + delta_x * delta_y * (1-delta_z) * out5
                 + delta_x * (1-delta_y) * delta_z * out6
                 + (1-delta_x) * delta_y * delta_z * out7
                 + delta_x * delta_y * delta_z * out8;


    float final_X_1, final_X_2, final_Y_1, final_Y_2, final_Z_1, final_Z_2;
    cudaMemcpy(&final_X_1, &d_X[X_index_1], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_X_2, &d_X[X_index_2], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_Y_1, &d_Y[Y_index_1], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_Y_2, &d_Y[Y_index_2], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_Z_1, &d_Z[Z_index_1], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_Z_2, &d_Z[Z_index_2], sizeof(float), cudaMemcpyDeviceToHost);

    // float *h_Z = (float *)malloc(h_SIZE_Z * sizeof(float));
    // cudaMemcpy(h_Z, d_Z, h_SIZE_Z * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i = 0 ; i< h_SIZE_Z;i++){
    // printf("Z = %f\n", h_Z[i]);}

    printf("X_index_1 = %d, X_index_2 = %d, Y_index_1 = %d, Y_index_2 = %d, Z_index_1 = %d, Z_index_2 = %d\n", 
            X_index_1, X_index_2, Y_index_1, Y_index_2, Z_index_1, Z_index_2);
    printf("1/2---对应的账户值是：%f, %f, %f, %f, %f, %f\n", 
            final_X_1, final_X_2, final_Y_1, final_Y_2, final_Z_1, final_Z_2);
    printf("相应的delta值是：%f, %f, %f\n", delta_x, delta_y, delta_z);
    printf("out1 = %f, out2 = %f, out3 = %f, out4 = %f, out5 = %f, out6 = %f, out7 = %f, out8 = %f, output = %f\n", 
            out1, out2, out3, out4, out5, out6, out7, out8, output);


    cudaFree(d_rng_states);

    return output;
}

void run(){//cuda3:0.0397, cuda 2: 0.0399;cuda 1: 0.0403；cuda0: 0.0405 ;//cuda #3: 0.03 female25  0.03(1200. 101.607758)
    float l = 0.039748f;
    printf("l = %f\n", l);

    init_global_config(
        0, 800, 7,
        0, 800, 3,
        0, 100, 3,
        0, 1,   2,
        0, 800, 3,
        0.15, 0.025, 0.05, 0.05, 0.2, 1000, 1, 100.0);

    init_global_XYZEW_V();

    init_texture_surface_object();

    // float output = compute_l2(l, trans_tau_np, 10);

    // reset_Vtp1();
    time_t start, end;
    time(&start);
    

    dim3 block(512);
    dim3 grid((h_sWEYZX + block.x - 1) / block.x);

    test_array_kernel<<<grid, block>>>(texObj0, texObj1);
    CUDA_CHECK(cudaGetLastError());     // launch
    CUDA_CHECK(cudaDeviceSynchronize()); // runtime


    float *h_results = (float *)malloc(h_sWEYZX * sizeof(float));
    cudaMemcpy(h_results, d_results, h_sWEYZX * sizeof(float), cudaMemcpyDeviceToHost);
    for(int idx = 0; idx < h_sWEYZX; idx++){

        int index_w = idx / h_sEYZX;
        int remainder = idx % h_sEYZX;
        int index_e = remainder / h_sYZX;
        remainder = remainder % h_sYZX;
        int index_y = remainder / h_sZX;
        remainder = remainder % h_sZX;
        int index_z = remainder / h_sX;
        int index_x = remainder % h_sX;
        if (index_w > 0){
            break;
        }
        if (true) {
            printf("W = %d, E = %d, Y = %d, Z = %d, X = %d, results[%d] = %f\n", 
                    index_w, index_e, index_y, index_z, index_x, idx, h_results[idx]);
        }
    }
    free(h_results);
    
    
    







    time(&end);
    printf("\n cpmputlel cost time = %f\n", difftime(end, start));

    clean_global_XYZEW_V();



}

void run2(){
    float l = 0.039748f;
    printf("l = %f\n", l);

    init_global_config(
        0, 800, 101,
        0, 800, 81,
        0, 100, 101,
        0, 1,   2,
        0, 800, 81,
        0.15, 0.025, 0.05, 0.05, 0.2, 1000, 1, 100.0);

    init_global_XYZEW_V();

    init_texture_surface_object();

    time_t start, end;
    time(&start);
    compute_l(l, trans_tau_np, 25);

    time(&end);

    printf("\n cpmputlel cost time = %f\n", difftime(end, start));

    clean_global_XYZEW_V();
}


int main() {

    run2();
    return 0;
}



















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






