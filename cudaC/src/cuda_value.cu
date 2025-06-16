#include "cuda_value.h"


// 生成随机数
__global__ void setup(curandStatePhilox4_32_10_t *state, unsigned long seed, int PATHS)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= PATHS) return;
    /* sequence=tid, offset=0 → 线程独立子流 */
    curand_init(seed, tid, 0, &state[tid]);
}

void init_random_state(curandStatePhilox4_32_10_t *d_state, unsigned long seed, int PATHS){
    cudaMalloc(&d_state,  PATHS*sizeof(*d_state));
    setup<<<(PATHS+1023)/1024,1024>>>(d_state, seed, PATHS);
    cudaDeviceSynchronize();
}



// 在 cuda_value.cu 文件开头定义全局变量
// __device__ __host__ float *d_X, *d_Y, *d_Z, *d_W, *d_V, *d_V_tp1;
// __device__ __host__ int *d_E;

__host__ __device__ int IDX_V(int x, int y, int z, int e) {
    // if(x*sYZEW + y*sZEW + z*sEW + e*sW + e >= SIZE_X * SIZE_Y * SIZE_Z * SIZE_E) {
    //     printf("Error: Index out of bounds\n");
    //     exit(1);
    // }
    
    return (x*sYZE + y*sZE + z*sE + e);
}

__host__ int h_IDX_V(int x, int y, int z, int e){
    return (x*h_sYZE + y*h_sZE + z*h_sE + e);
}

// cuda_value.cu
int h_MIN_X, h_MIN_Y, h_MIN_Z, h_MIN_W;
int h_MAX_X, h_MAX_Y, h_MAX_Z, h_MAX_W;
int h_SIZE_X, h_SIZE_Y, h_SIZE_Z, h_SIZE_E, h_SIZE_W;
int h_sXYZEW, h_sYZEW, h_sZEW, h_sEW, h_sW;
int h_sXYZE, h_sYZE, h_sZE, h_sE;
float h_SCALE_TO_INT_X, h_SCALE_TO_INT_Y, h_SCALE_TO_INT_Z;

float *d_X, *d_Y, *d_Z, *d_W, *d_V, *d_V_tp1, *d_results;
int *d_E;

void init_global_config(
    int min_X, int max_X, int size_X,
    int min_Y, int max_Y, int size_Y,
    int min_Z, int max_Z, int size_Z,
    int min_E, int max_E, int size_E,
    int min_W, int max_W, int size_W
){
    h_MIN_X = min_X;
    h_MAX_X = max_X;
    h_SIZE_X = size_X;
    h_MIN_Y = min_Y;
    h_MAX_Y = max_Y;
    h_SIZE_Y = size_Y;
    h_MIN_Z = min_Z;
    h_MAX_Z = max_Z;
    h_SIZE_Z = size_Z;
    h_SIZE_E = size_E;
    h_MIN_W = min_W;
    h_MAX_W = max_W;
    h_SIZE_W = size_W;

    h_sXYZEW = size_X * size_Y * size_Z * size_E * size_W;
    h_sYZEW = size_Y * size_Z * size_E * size_W;
    h_sZEW = size_Z * size_E * size_W;
    h_sEW = size_E * size_W;
    h_sW = size_W;

    h_sXYZE = size_X * size_Y * size_Z * size_E;   
    h_sYZE = size_Y * size_Z * size_E;
    h_sZE = size_Z * size_E;
    h_sE = size_E;

    h_SCALE_TO_INT_X = (float)(size_X-1) / (max_X - min_X);
    h_SCALE_TO_INT_Y = (float)(size_Y-1) / (max_Y - min_Y);
    h_SCALE_TO_INT_Z = (float)(size_Z-1) / (max_Z - min_Z);

    cudaMemcpyToSymbolAsync(d_MIN_X, &h_MIN_X, sizeof(int));
    cudaMemcpyToSymbolAsync(d_MAX_X, &h_MAX_X, sizeof(int));
    cudaMemcpyToSymbolAsync(d_SIZE_X, &h_SIZE_X, sizeof(int));
    cudaMemcpyToSymbolAsync(d_MIN_Y, &h_MIN_Y, sizeof(int));
    cudaMemcpyToSymbolAsync(d_MAX_Y, &h_MAX_Y, sizeof(int));
    cudaMemcpyToSymbolAsync(d_SIZE_Y, &h_SIZE_Y, sizeof(int));
    cudaMemcpyToSymbolAsync(d_MIN_Z, &h_MIN_Z, sizeof(int));
    cudaMemcpyToSymbolAsync(d_MAX_Z, &h_MAX_Z, sizeof(int));
    cudaMemcpyToSymbolAsync(d_SIZE_Z, &h_SIZE_Z, sizeof(int));
    cudaMemcpyToSymbolAsync(d_MIN_W, &h_MIN_W, sizeof(int));
    cudaMemcpyToSymbolAsync(d_MAX_W, &h_MAX_W, sizeof(int));
    cudaMemcpyToSymbolAsync(d_SIZE_W, &h_SIZE_W, sizeof(int));

    cudaMemcpyToSymbolAsync(d_sXYZEW, &h_sXYZEW, sizeof(int));
    cudaMemcpyToSymbolAsync(d_sYZEW, &h_sYZEW, sizeof(int));
    cudaMemcpyToSymbolAsync(d_sZEW, &h_sZEW, sizeof(int));
    cudaMemcpyToSymbolAsync(d_sEW, &h_sEW, sizeof(int));
    cudaMemcpyToSymbolAsync(d_sW, &h_sW, sizeof(int));

    cudaMemcpyToSymbolAsync(d_sXYZE, &h_sXYZE, sizeof(int));
    cudaMemcpyToSymbolAsync(d_sYZE, &h_sYZE, sizeof(int));
    cudaMemcpyToSymbolAsync(d_sZE, &h_sZE, sizeof(int));
    cudaMemcpyToSymbolAsync(d_sE, &h_sE, sizeof(int));

    cudaMemcpyToSymbolAsync(d_SCALE_TO_INT_X, &h_SCALE_TO_INT_X, sizeof(float));
    cudaMemcpyToSymbolAsync(d_SCALE_TO_INT_Y, &h_SCALE_TO_INT_Y, sizeof(float));
    cudaMemcpyToSymbolAsync(d_SCALE_TO_INT_Z, &h_SCALE_TO_INT_Z, sizeof(float));


    // 初始化XYZEW_V
    float *h_X = (float *)malloc(h_SIZE_X * sizeof(float));
    float *h_Y = (float *)malloc(h_SIZE_Y * sizeof(float));
    float *h_Z = (float *)malloc(h_SIZE_Z * sizeof(float));
    int   *h_E = (int   *)malloc(h_SIZE_E * sizeof(int));
    float *h_W = (float *)malloc(h_SIZE_W * sizeof(float));
    float *h_V = (float *)malloc(h_sXYZE * sizeof(float));
    
    for (int i = 0; i < h_SIZE_X; i++) {
        h_X[i] = h_MIN_X + (h_MAX_X - h_MIN_X) * i / (h_SIZE_X - 1);
    }
    for (int i = 0; i < h_SIZE_Y; i++) {
        h_Y[i] = h_MIN_Y + (h_MAX_Y - h_MIN_Y) * i / (h_SIZE_Y - 1);
    }
    for (int i = 0; i < h_SIZE_Z; i++) {
        h_Z[i] = h_MIN_Z + (h_MAX_Z - h_MIN_Z) * i / (h_SIZE_Z - 1);
    }
    for (int i = 0; i < h_SIZE_E; i++) {
        h_E[i] = i;
    }
    for (int i = 0; i < h_SIZE_W; i++) {
        h_W[i] = h_MIN_W + (h_MAX_W - h_MIN_W) * i / (h_SIZE_W - 1);
    }

    // 初始化 V 数组
    printf("Initializing V array...\n");
    for (int x = 0; x < h_SIZE_X; x++) {
        for (int y = 0; y < h_SIZE_Y; y++) {
            for (int z = 0; z < h_SIZE_Z; z++) {
                float min_ZY = fminf(h_Z[z], h_Y[y]);
                float term = (h_Y[y] <= min_ZY) ? 
                            h_Y[y] : 
                            h_Y[y] - A1 * (h_Y[y] - min_ZY);
                float result = fmaxf(h_X[x], term);
                
                // 对 E 的两个维度都赋值
                h_V[h_IDX_V(x, y, z, 0)] = result;
                h_V[h_IDX_V(x, y, z, 1)] = result;
            }
        }
    }

    printf("V array initialized\n");

    // 分配设备内存
    cudaMalloc(&d_X, h_SIZE_X * sizeof(float));
    cudaMalloc(&d_Y, h_SIZE_Y * sizeof(float));
    cudaMalloc(&d_Z, h_SIZE_Z * sizeof(float)); 
    cudaMalloc(&d_E, h_SIZE_E * sizeof(int));
    cudaMalloc(&d_W, h_SIZE_W * sizeof(float));
    cudaMalloc(&d_V, h_sXYZE * sizeof(float));
    cudaMalloc(&d_V_tp1, h_sXYZE * sizeof(float));
    cudaMalloc(&d_results, h_sXYZEW * sizeof(float));
    printf("cudamalloc done\n");
    // 检查内存分配是否成功
    // if (!d_X || !d_Y || !d_Z || !d_E || !d_W || !d_V || !d_V_tp1 || !d_results) {
    //     printf("Error: Failed to allocate device memory\n");
    //     clean_global_XYZEW_V();
    //     exit(1);
    // }

    

    // 复制数据到设备
    cudaMemcpy(d_X, h_X, h_SIZE_X * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, h_SIZE_Y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, h_Z, h_SIZE_Z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, h_E, h_SIZE_E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, h_SIZE_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, h_sXYZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_tp1, h_V, h_sXYZE * sizeof(float), cudaMemcpyHostToDevice);
    

    // 将主机端指针值复制到设备端全局变量
    cudaMemcpyToSymbolAsync(d_d_X, &d_X, sizeof(float*));
    cudaMemcpyToSymbolAsync(d_d_Y, &d_Y, sizeof(float*));
    cudaMemcpyToSymbolAsync(d_d_Z, &d_Z, sizeof(float*));
    cudaMemcpyToSymbolAsync(d_d_E, &d_E, sizeof(int*));
    cudaMemcpyToSymbolAsync(d_d_W, &d_W, sizeof(float*));
    cudaMemcpyToSymbolAsync(d_d_V, &d_V, sizeof(float*));
    cudaMemcpyToSymbolAsync(d_d_V_tp1, &d_V_tp1, sizeof(float*));
    cudaMemcpyToSymbolAsync(d_d_results, &d_results, sizeof(float*));

    printf("init_global_XYZEW_V is done\n");



    // 释放主机内存
    free(h_X);
    free(h_Y);
    free(h_Z);
    free(h_E);
    free(h_W);
    free(h_V);

}

void init_global_XYZEW_V() {
    // 分配主机内存
    float *h_X = (float *)malloc(SIZE_X * sizeof(float));
    float *h_Y = (float *)malloc(SIZE_Y * sizeof(float));
    float *h_Z = (float *)malloc(SIZE_Z * sizeof(float));
    int   *h_E = (int   *)malloc(SIZE_E * sizeof(int));
    float *h_W = (float *)malloc(SIZE_W * sizeof(float));
    float *h_V = (float *)malloc(SIZE_X * SIZE_Y * SIZE_Z * SIZE_E *sizeof(float));
    
    // 初始化 X, Y, Z, W 数组
    for (int i = 0; i < SIZE_X; i++) {
        h_X[i] = MIN_XYZ + (MAX_X - MIN_XYZ) * i / (SIZE_X - 1);
    }
    for (int i = 0; i < SIZE_Y; i++) {
        h_Y[i] = MIN_XYZ + (MAX_Y - MIN_XYZ) * i / (SIZE_Y - 1);
    }
    for (int i = 0; i < SIZE_Z; i++) {
        h_Z[i] = MIN_XYZ + (MAX_Z - MIN_XYZ) * i / (SIZE_Z - 1);
    }
    for (int i = 0; i < SIZE_E; i++) {
        h_E[i] = i;
    }
    for (int i = 0; i < SIZE_W; i++) {
        h_W[i] = MIN_XYZ + (MAX_W - MIN_XYZ) * i / (SIZE_W - 1);
    }

    // 初始化 V 数组
    printf("Initializing V array...\n");
    for (int x = 0; x < SIZE_X; x++) {
        for (int y = 0; y < SIZE_Y; y++) {
            for (int z = 0; z < SIZE_Z; z++) {
                float min_ZY = fminf(h_Z[z], h_Y[y]);
                float term = (h_Y[y] <= min_ZY) ? 
                            h_Y[y] : 
                            h_Y[y] - A1 * (h_Y[y] - min_ZY);
                float result = fmaxf(h_X[x], term);
                
                // 对 E 的两个维度都赋值
                h_V[IDX_V(x, y, z, 0)] = result;
                h_V[IDX_V(x, y, z, 1)] = result;
            }
        }
    }

    // 分配设备内存
    cudaMalloc(&d_X, SIZE_X * sizeof(float));
    cudaMalloc(&d_Y, SIZE_Y * sizeof(float));
    cudaMalloc(&d_Z, SIZE_Z * sizeof(float));
    cudaMalloc(&d_E, SIZE_E * sizeof(int));
    cudaMalloc(&d_W, SIZE_W * sizeof(float));
    cudaMalloc(&d_V, SIZE_X * SIZE_Y * SIZE_Z * SIZE_E * sizeof(float));
    cudaMalloc(&d_V_tp1, SIZE_X * SIZE_Y * SIZE_Z * SIZE_E * sizeof(float));
    cudaMalloc(&d_results, SIZE_X * SIZE_Y * SIZE_Z * SIZE_E * SIZE_W * sizeof(float));

    // 检查内存分配是否成功
    if (!d_X || !d_Y || !d_Z || !d_E || !d_W || !d_V || !d_V_tp1 || !d_results) {
        printf("Error: Failed to allocate device memory\n");
        clean_global_XYZEW_V();
        exit(1);
    }

    // 复制数据到设备
    cudaMemcpy(d_X, h_X, SIZE_X * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, SIZE_Y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, h_Z, SIZE_Z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, h_E, SIZE_E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, SIZE_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, SIZE_X * SIZE_Y * SIZE_Z * SIZE_E * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_tp1, h_V, SIZE_X * SIZE_Y * SIZE_Z * SIZE_E * sizeof(float), cudaMemcpyHostToDevice);


    // 将主机端指针值复制到设备端全局变量
    cudaMemcpyToSymbolAsync(d_d_X, &d_X, sizeof(float*));
    cudaMemcpyToSymbolAsync(d_d_Y, &d_Y, sizeof(float*));
    cudaMemcpyToSymbolAsync(d_d_Z, &d_Z, sizeof(float*));
    cudaMemcpyToSymbolAsync(d_d_E, &d_E, sizeof(int*));
    cudaMemcpyToSymbolAsync(d_d_W, &d_W, sizeof(float*));
    cudaMemcpyToSymbolAsync(d_d_V, &d_V, sizeof(float*));
    cudaMemcpyToSymbolAsync(d_d_V_tp1, &d_V_tp1, sizeof(float*));
    cudaMemcpyToSymbolAsync(d_d_results, &d_results, sizeof(float*));

    // 检查内存复制是否成功
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
        clean_global_XYZEW_V();
        exit(1);
    }




    // 释放主机内存
    free(h_X);
    free(h_Y);
    free(h_Z);
    free(h_E);
    free(h_W);
    free(h_V);
}

// 清理函数
void clean_global_XYZEW_V() {
    if (d_X) cudaFree(d_X);
    if (d_Y) cudaFree(d_Y);
    if (d_Z) cudaFree(d_Z);
    if (d_E) cudaFree(d_E);
    if (d_W) cudaFree(d_W);
    if (d_V) cudaFree(d_V);
    if (d_V_tp1) cudaFree(d_V_tp1);
    if (d_results) cudaFree(d_results);

    d_X = nullptr;
    d_Y = nullptr;
    d_Z = nullptr;
    d_E = nullptr;
    d_W = nullptr;
    d_V = nullptr;
    d_V_tp1 = nullptr;
    d_results = nullptr;

    printf("clean_global_XYZEW_V is done\n");
}   


// 一轮计算后重置Vtp1
void reset_Vtp1() {
    cudaMemcpy(d_V_tp1, d_V, sXYZE * sizeof(float), cudaMemcpyDeviceToDevice);
}




// 查表函数
__device__ float lookup_V(float X, float Y, float Z, int E) {
    float scale_to_int = (float)SIZE_X / (MAX_X - MIN_XYZ);
    
    int X_int = (int)floorf((X - MIN_XYZ) * scale_to_int);
    int Y_int = (int)floorf((Y - MIN_XYZ) * scale_to_int);
    int Z_int = (int)floorf((Z - MIN_XYZ) * scale_to_int);
    int E_int = E;
    
    return d_d_V_tp1[IDX_V(X_int, Y_int, Z_int, E_int)];
}



// 设备函数实现
__device__ float monte_carlo_simulation(float XmW, float Y_tp1, float Z_tp1, int E_tp1, float P_tau_tp1, float P_tau_gep_tp1, float l, curandStatePhilox4_32_10_t * rng_states, int idx) {
    float d_temp = 0.0f;
    curandStatePhilox4_32_10_t s = rng_states[idx];
    
    // 预计算常用值
    const float exp_term = expf((MU - l - 0.5f * SIGMA * SIGMA) * DELTA_T);
    const float sqrt_delta_t = sqrtf(DELTA_T);
    const float discount_factor = expf(-R * DELTA_T);
    
    // Monte Carlo 模拟
    for (int i = 0; i < MOTECALO_NUMS; i++) {
        // 生成随机数
        float random = curand_normal(&s);
        
        // d_temp += 1000 * random;

        // 计算 X(t+1)
        float X_tp1 = XmW * exp_term * expf(SIGMA * sqrt_delta_t * random);
        X_tp1 = fminf(X_tp1, MAX_X);
        
        // 查找值函数
        float V_tp1 = lookup_V(X_tp1, Y_tp1, Z_tp1, E_tp1);
        
        // 累加结果
        d_temp += discount_factor * (P_tau_tp1 * fmaxf(X_tp1, Y_tp1) + 
                                   P_tau_gep_tp1 * V_tp1);
    }

    rng_states[idx] = s;
    
    return d_temp ;
}

// XYZEW kernel 实现
__global__ void XYZEW_kernel(int offset, int t, curandStatePhilox4_32_10_t *rng_states, float l, float a3, float P_tau_gep_tp1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx >= sXYZEW) return;

    // 计算索引
    int index_x = idx / sYZEW;
    int remainder = idx % sYZEW;
    int index_y = remainder / sZEW;
    remainder = remainder % sZEW;
    int index_z = remainder / sEW;
    remainder = remainder % sEW;
    int index_e = remainder / sW;
    int index_w = remainder % sW;

    // 获取值
    float X = d_d_X[index_x];
    float Y = d_d_Y[index_y];
    float Z = d_d_Z[index_z];
    int E = d_d_E[index_e];
    float W = d_d_W[index_w];

    float min_ZYt = fminf(Z, Y);
    float Y_tp1, Z_tp1;

    int E_tp1 = 1 * (E + W == 0);





    // 优化
    // // ---------- 预先算好共用量 ----------
    const float invX  = __frcp_rn(X);               // 1/X  (更省时钟)
    const float XmW   = fmaxf(X - W, 0.0f);         // max(X-W,0)
    const bool  wz    = (W == 0);
    const bool  ez    = (E_tp1 == 0);
    const bool  wle   = (W <= min_ZYt);

    // ---------- path-specific候选值 ----------
    const float Y00 = (1.0f + A2) * fmaxf(X,        Y);          // W==0 && E==0
    const float Z00 = (1.0f + A2) * fmaxf(a3 * X,   Z);

    const float Y01 =                fmaxf(X,        Y);          // W==0 && E>0
    const float Z01 =                fmaxf(a3 * X,   Z);

    const float Y10 = fmaxf(XmW,      Y - W);                     // W>0 && W<=min_ZYt
    const float Z10 = fmaxf(a3 * XmW, Z);

    const float t111    = fminf(Y - W,   Y * invX * XmW);            // W>0 && W>min_ZYt
    const float Y11 = fmaxf(XmW,      t111);
    const float Z11 = fmaxf(a3 * XmW, Z * invX * XmW);

    // ---------- 4 个掩码 ----------
    const float m00 =  wz &  ez;          // W==0 &&  E==0
    const float m01 =  wz & !ez;          // W==0 &&  E>0
    const float m10 = !wz &  wle;         // W>0 &&  W<=min_ZYt
    const float m11 = !wz & !wle;         // W>0 &&  W> min_ZYt

    // ---------- 混合得到最终结果 ----------
    Y_tp1 = m00 * Y00 + m01 * Y01 + m10 * Y10 + m11 * Y11;
    Z_tp1 = m00 * Z00 + m01 * Z01 + m10 * Z10 + m11 * Z11;

        // P_tau_tp1 = d_P_tau[0] # 这个是P(tau=t+1)时刻的值
        // P_tau_gep_tp1 = d_P_tau[1] # 这个是P(tau>=t+1)时刻的值


    float P_tau_tp1 = 1 - P_tau_gep_tp1;

 

    // //Monte Carlo 模拟
    float d_temp = monte_carlo_simulation(
        XmW, Y_tp1, Z_tp1, E_tp1,
        P_tau_tp1, P_tau_gep_tp1,
        l, rng_states, idx
    );



    // 优化代码
    // ─── 仅用 3 条浮点指令 + 1 条乘 fWt *= (t != 0) ──────────
    float fWt = W - A1 * fmaxf(W - min_ZYt, 0.0f);   // ← 已同时覆盖两种情况
    fWt       *= (t != 0);                           // t==0 → 置 0

    
    // 存储结果
    d_d_results[idx] = d_temp / MOTECALO_NUMS + fWt;
    // d_results[idx] = d_temp;
}

// V_tp1 kernel 实现
__global__ void V_tp1_kernel(int offset, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx >= SIZE_X * SIZE_Y * SIZE_Z * SIZE_E) return;

    // 计算索引
    int index_x = idx / sYZE;
    int remainder = idx % sYZE;
    int index_y = remainder / sZE;
    remainder = remainder % sZE;
    int index_z = remainder / sE;
    // int index_e = remainder % sE;

    float X = d_d_X[index_x];
    float Y = d_d_Y[index_y];
    float Z = d_d_Z[index_z];
    // int E = d_d_E[index_e];

    int W_index = idx * SIZE_W;
    float max_w = d_d_results[W_index];

    if (t == 0) {
        d_d_V_tp1[idx] = max_w;
        return;
    }

    // 查找最大值
    for (int i = 0; i < SIZE_W; i++) {

        if (Y >= d_d_W[i]) {
            float current = d_d_results[W_index + i];
            if (current > max_w) {
                max_w = current;
            }
        }
    }

    d_d_V_tp1[idx] = fmaxf(fmaxf(Y - A1 * fmaxf((Y - fminf(Z, Y)), 0.0f), X), max_w);
}



