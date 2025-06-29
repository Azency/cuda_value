#include "cuda_value.h"
// #include "config.h"

float h_MIN_X, h_MIN_Y, h_MIN_Z, h_MIN_W;
float h_MAX_X, h_MAX_Y, h_MAX_Z, h_MAX_W;
int h_SIZE_X, h_SIZE_Y, h_SIZE_Z, h_SIZE_E, h_SIZE_W;
int h_sXYZEW, h_sYZEW, h_sZEW, h_sEW, h_sW;
int h_sXYZE, h_sYZE, h_sZE, h_sE;
float h_SCALE_TO_INT_X, h_SCALE_TO_INT_Y, h_SCALE_TO_INT_Z;

__constant__ float d_MIN_X, d_MIN_Y, d_MIN_Z, d_MIN_W;
__constant__ float d_MAX_X, d_MAX_Y, d_MAX_Z, d_MAX_W;
__constant__ int d_SIZE_X, d_SIZE_Y, d_SIZE_Z, d_SIZE_E, d_SIZE_W;
__constant__ int d_sXYZEW, d_sYZEW, d_sZEW, d_sEW, d_sW;
__constant__ int d_sXYZE, d_sYZE, d_sZE, d_sE;
__constant__ float d_SCALE_TO_INT_X, d_SCALE_TO_INT_Y, d_SCALE_TO_INT_Z;


float h_A1,h_P, h_INITIAL_INVESTMENT, h_DELTA_T;
__constant__ float d_A1, d_A2, d_R, d_MU, d_SIGMA, d_P, d_INITIAL_INVESTMENT, d_DELTA_T;
__constant__ int d_MOTECALO_NUMS;

float *d_X, *d_Y, *d_Z, *d_W, *d_V, *d_V_tp1, *d_results;
int *d_E;

__constant__ float *d_d_X, *d_d_Y, *d_d_Z, *d_d_W, *d_d_V, *d_d_V_tp1, *d_d_results;
__constant__ int *d_d_E;

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

__device__ int IDX_V(int x, int y, int z, int e) {
    // if(x*sYZEW + y*sZEW + z*sEW + e*sW + e >= SIZE_X * SIZE_Y * SIZE_Z * SIZE_E) {
    //     printf("Error: Index out of bounds\n");
    //     exit(1);
    // }
    
    return (x*d_sYZE + y*d_sZE + z*d_sE + e);
}

__host__ int h_IDX_V(int x, int y, int z, int e){
    return (x*h_sYZE + y*h_sZE + z*h_sE + e);
}


void init_global_config(
    float min_X, float max_X, int size_X,
    float min_Y, float max_Y, int size_Y,
    float min_Z, float max_Z, int size_Z,
    int min_E, int max_E, int size_E,
    float min_W, float max_W, int size_W,
    float a1, float a2, float r, float mu, float sigma, int motecalo_nums, float p, float initial_investment
){
    
    h_A1 = a1;
    cudaMemcpyToSymbolAsync(d_A1, &h_A1, sizeof(float));
    float h_A2 = a2;
    cudaMemcpyToSymbolAsync(d_A2, &h_A2, sizeof(float));
    float h_R = r;
    cudaMemcpyToSymbolAsync(d_R, &h_R, sizeof(float));
    float h_MU = mu;
    cudaMemcpyToSymbolAsync(d_MU, &h_MU, sizeof(float));
    float h_SIGMA = sigma;
    cudaMemcpyToSymbolAsync(d_SIGMA, &h_SIGMA, sizeof(float));
    int h_MOTECALO_NUMS = motecalo_nums;
    cudaMemcpyToSymbolAsync(d_MOTECALO_NUMS, &h_MOTECALO_NUMS, sizeof(int));
    h_P = p;
    cudaMemcpyToSymbolAsync(d_P, &h_P, sizeof(float));
    h_INITIAL_INVESTMENT = initial_investment;
    cudaMemcpyToSymbolAsync(d_INITIAL_INVESTMENT, &h_INITIAL_INVESTMENT, sizeof(float));
    h_DELTA_T = 1.0f/h_P;
    cudaMemcpyToSymbolAsync(d_DELTA_T, &h_DELTA_T, sizeof(float));
    
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

    cudaMemcpyToSymbolAsync(d_MIN_X, &h_MIN_X, sizeof(float));
    cudaMemcpyToSymbolAsync(d_MAX_X, &h_MAX_X, sizeof(float));
    cudaMemcpyToSymbolAsync(d_MIN_Y, &h_MIN_Y, sizeof(float));
    cudaMemcpyToSymbolAsync(d_MAX_Y, &h_MAX_Y, sizeof(float));
    cudaMemcpyToSymbolAsync(d_MIN_Z, &h_MIN_Z, sizeof(float));
    cudaMemcpyToSymbolAsync(d_MAX_Z, &h_MAX_Z, sizeof(float));
    cudaMemcpyToSymbolAsync(d_MIN_W, &h_MIN_W, sizeof(float));
    cudaMemcpyToSymbolAsync(d_MAX_W, &h_MAX_W, sizeof(float));
    
    cudaMemcpyToSymbolAsync(d_SIZE_X, &h_SIZE_X, sizeof(int)); 
    cudaMemcpyToSymbolAsync(d_SIZE_Y, &h_SIZE_Y, sizeof(int));
    cudaMemcpyToSymbolAsync(d_SIZE_Z, &h_SIZE_Z, sizeof(int));   
    cudaMemcpyToSymbolAsync(d_SIZE_E, &h_SIZE_E, sizeof(int));
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


 

}

void init_global_XYZEW_V() {
   // 初始化XYZEW_V
    float *h_X = (float *)malloc(h_SIZE_X * sizeof(float));
    float *h_Y = (float *)malloc(h_SIZE_Y * sizeof(float));
    float *h_Z = (float *)malloc(h_SIZE_Z * sizeof(float));
    int   *h_E = (int   *)malloc(h_SIZE_E * sizeof(int));
    float *h_W = (float *)malloc(h_SIZE_W * sizeof(float));
    float *h_V = (float *)malloc(h_sXYZE * sizeof(float));
    
    for (int i = 0; i < h_SIZE_X; i++) {
        h_X[i] = h_MIN_X + float(h_MAX_X - h_MIN_X) * i / (h_SIZE_X - 1);
    }
    for (int i = 0; i < h_SIZE_Y; i++) {
        h_Y[i] = h_MIN_Y + float(h_MAX_Y - h_MIN_Y) * i / (h_SIZE_Y - 1);
    }
    for (int i = 0; i < h_SIZE_Z; i++) {
        h_Z[i] = h_MIN_Z + float(h_MAX_Z - h_MIN_Z) * i / (h_SIZE_Z - 1);
    }
    for (int i = 0; i < h_SIZE_E; i++) {
        h_E[i] = i;
    }
    for (int i = 0; i < h_SIZE_W; i++) {
        h_W[i] = h_MIN_W + float(h_MAX_W - h_MIN_W) * i / (h_SIZE_W - 1);
    }

    // 初始化 V 数组
    printf("Initializing V array...\n");
    for (int x = 0; x < h_SIZE_X; x++) {
        for (int y = 0; y < h_SIZE_Y; y++) {
            for (int z = 0; z < h_SIZE_Z; z++) {
                float min_ZY = fminf(h_Z[z], h_Y[y]);
                float term = (h_Y[y] <= min_ZY) ? 
                            h_Y[y] : 
                            h_Y[y] - h_A1 * (h_Y[y] - min_ZY);
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
    cudaMemcpy(d_V_tp1, d_V, h_sXYZE * sizeof(float), cudaMemcpyDeviceToDevice);
}




// 查表函数
__device__ float lookup_V(float X, float Y, float Z, int E) {
    int E_int = E;
    int X_down = (int)floorf((X - d_MIN_X) * d_SCALE_TO_INT_X);
    int Y_down = (int)floorf((Y - d_MIN_Y) * d_SCALE_TO_INT_Y);
    int Z_down = (int)floorf((Z - d_MIN_Z) * d_SCALE_TO_INT_Z);
    int X_up   = fminf(X_down + 1, d_SIZE_X - 1);
    int Y_up   = fminf(Y_down + 1, d_SIZE_Y - 1);
    int Z_up   = fminf(Z_down + 1, d_SIZE_Z - 1);

    float dx   = (X - d_d_X[X_down]) * d_SCALE_TO_INT_X;
    float dy   = (Y - d_d_Y[Y_down]) * d_SCALE_TO_INT_Y; 
    float dz   = (Z - d_d_Z[Z_down]) * d_SCALE_TO_INT_Z;

    // float res = d_d_V_tp1[IDX_V(X_down, Y_down, Z_down, E_int)] + d_d_V_tp1[IDX_V(X_up, Y_down, Z_down, E_int)] + d_d_V_tp1[IDX_V(X_down, Y_up, Z_down, E_int)] + d_d_V_tp1[IDX_V(X_up, Y_up, Z_down, E_int)] + d_d_V_tp1[IDX_V(X_down, Y_down, Z_up, E_int)] + d_d_V_tp1[IDX_V(X_up, Y_down, Z_up, E_int)] + d_d_V_tp1[IDX_V(X_down, Y_up, Z_up, E_int)] + d_d_V_tp1[IDX_V(X_up, Y_up, Z_up, E_int)];
    // res = res / 8;

    float res = (1 - dx) * (1 - dy) * (1 - dz) * d_d_V_tp1[IDX_V(X_down, Y_down, Z_down, E_int)] + 
                dx * (1 - dy) * (1 - dz) * d_d_V_tp1[IDX_V(X_up, Y_down, Z_down, E_int)] + 
                (1 - dx) * dy * (1 - dz) * d_d_V_tp1[IDX_V(X_down, Y_up, Z_down, E_int)] + 
                dx * dy * (1 - dz) * d_d_V_tp1[IDX_V(X_up, Y_up, Z_down, E_int)] + 
                (1 - dx) * (1 - dy) * dz * d_d_V_tp1[IDX_V(X_down, Y_down, Z_up, E_int)] + 
                dx * (1 - dy) * dz * d_d_V_tp1[IDX_V(X_up, Y_down, Z_up, E_int)] + 
                (1 - dx) * dy * dz * d_d_V_tp1[IDX_V(X_down, Y_up, Z_up, E_int)] + 
                dx * dy * dz * d_d_V_tp1[IDX_V(X_up, Y_up, Z_up, E_int)];


    // float V000 = d_d_V_tp1[IDX_V(X_down, Y_down, Z_down, E_int)];
    // float V100 = d_d_V_tp1[IDX_V(X_up, Y_down, Z_down, E_int)];
    // float V010 = d_d_V_tp1[IDX_V(X_down, Y_up, Z_down, E_int)];
    // float V110 = d_d_V_tp1[IDX_V(X_up, Y_up, Z_down, E_int)];
    // float V001 = d_d_V_tp1[IDX_V(X_down, Y_down, Z_up, E_int)];
    // float V101 = d_d_V_tp1[IDX_V(X_up, Y_down, Z_up, E_int)];
    // float V011 = d_d_V_tp1[IDX_V(X_down, Y_up, Z_up, E_int)];
    // float V111 = d_d_V_tp1[IDX_V(X_up, Y_up, Z_up, E_int)];

    // float res = (1 - dx) * (1 - dy) * (1 - dz) * V000 + dx* (1 - dy) * (1 - dz) * V100 + 
    //             (1 - dx) * dy       * (1 - dz) * V010 + 
    //             dx * dy  * (1 - dz) * V110 + 
    //             (1 - dx) * (1 - dy) * dz       * V001 + 
    //             dx       * (1 - dy) * dz       * V101 + 
    //             (1 - dx) * dy       * dz       * V011 + 
    //             dx       * dy       * dz       * V111;

    // float res = V000 + V100 + V010 + V110 + V001 + V101 + V011 + V111;

    return res;


    // int X_down = (int)floorf((X - d_MIN_X) * d_SCALE_TO_INT_X);
    // int Y_down = (int)floorf((Y - d_MIN_Y) * d_SCALE_TO_INT_Y);
    // int Z_down = (int)floorf((Z - d_MIN_Z) * d_SCALE_TO_INT_Z);

    // return d_d_V_tp1[IDX_V(X_down, Y_down, Z_down, E)];
}



// 设备函数实现
__device__ float monte_carlo_simulation(float XmW, float Y_tp1, float Z_tp1, int E_tp1, float P_tau_tp1, float P_tau_gep_tp1, float l, curandStatePhilox4_32_10_t * rng_states, int idx) {
    float d_temp = 0.0f;
    curandStatePhilox4_32_10_t s = rng_states[idx];
    
    // 预计算常用值
    const float exp_term = expf((d_MU - l - 0.5f * d_SIGMA * d_SIGMA) * d_DELTA_T);
    const float sqrt_delta_t = sqrtf(d_DELTA_T);
    const float discount_factor = expf(-d_R * d_DELTA_T);
    
    // Monte Carlo 模拟
    for (int i = 0; i < d_MOTECALO_NUMS; i++) {
        // 生成随机数
        float random = curand_normal(&s);
        
        // d_temp += 1000 * random;

        // 计算 X(t+1)
        float X_tp1 = XmW * exp_term * expf(d_SIGMA * sqrt_delta_t * random);
        X_tp1 = fminf(X_tp1, d_MAX_X);
        
        // 查找值函数
        float V_tp1 = lookup_V(X_tp1, Y_tp1, Z_tp1, E_tp1);
        
        // 累加结果
        d_temp += discount_factor * (P_tau_tp1 * fmaxf(X_tp1, Y_tp1) + 
                                   P_tau_gep_tp1 * V_tp1);
    }

    rng_states[idx] = s;
    
    return d_temp ;
}


__global__ void monte_carlo_simulation_kernel(
    float XmW,           // X - W
    float Y_tp1,         // Y(t+1)
    float Z_tp1,         // Z(t+1)
    int E_tp1,           // E(t+1)
    float P_tau_tp1,     // P(tau=t+1)
    float P_tau_gep_tp1, // P(tau>=t+1)
    float l,             // 费用率
    curandStatePhilox4_32_10_t * rng_states, // 随机数状态
    int results_idx              // 线程索引
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_MOTECALO_NUMS) return;

    // float d_temp = 0.0f;
    // curandStatePhilox4_32_10_t s = rng_states[results_idx];

    // const float exp_term = expf((d_MU - l - 0.5f * d_SIGMA * d_SIGMA) * d_DELTA_T);
    // const float sqrt_delta_t = sqrtf(d_DELTA_T);
    // const float discount_factor = expf(-d_R * d_DELTA_T);

    // // 生成随机数
    // float random = curand_normal(&s);
    
    // // d_temp += 1000 * random;

    // // 计算 X(t+1)
    // float X_tp1 = XmW * exp_term * expf(d_SIGMA * sqrt_delta_t * random);
    // X_tp1 = fminf(X_tp1, d_MAX_X);
    
    // // 查找值函数
    // float V_tp1 = lookup_V(X_tp1, Y_tp1, Z_tp1, E_tp1); 

    // d_temp += discount_factor * (P_tau_tp1 * fmaxf(X_tp1, Y_tp1) + 
    //                                P_tau_gep_tp1 * V_tp1);


    atomicAdd(&d_d_results[results_idx], 0.125);

    // rng_states[results_idx] = s;

}

// XYZEW kernel 实现
__global__ void XYZEW_kernel2(int offset, int t, curandStatePhilox4_32_10_t *rng_states, float l, float a3, float P_tau_gep_tp1) {
    int idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    if (idx >= d_sXYZEW) return;

    // 计算索引
    int index_x = idx / d_sYZEW;
    int remainder = idx % d_sYZEW;
    int index_y = remainder / d_sZEW;
    remainder = remainder % d_sZEW;
    int index_z = remainder / d_sEW;
    remainder = remainder % d_sEW;
    int index_e = remainder / d_sW;
    int index_w = remainder % d_sW;

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
    const float Y00 = (1.0f + d_A2) * fmaxf(X,        Y);          // W==0 && E==0
    const float Z00 = (1.0f + d_A2) * fmaxf(a3 * X,   Z);

    const float Y01 = fmaxf(Y, XmW);          // W==0 && E>0
    const float Z01 = fmaxf(Z, a3 * XmW);

    const float Y10 = fmaxf(Y - W, XmW);                     // W>0 && W<=min_ZYt
    const float Z10 = fmaxf(Z, a3 * XmW);

    const float t11 = fminf(Y - W,   Y * invX * XmW);            // W>0 && W>min_ZYt
    const float Y11 = fmaxf(XmW,      t11);
    const float Z11 = fmaxf(a3 * XmW, Z * invX * XmW);

    // ---------- 4 个掩码 ----------
    const float m00 =  wz &  ez;          // W==0 &&  E==0
    const float m01 =  wz & !ez;          // W==0 &&  E>0
    const float m10 = !wz &  wle;         // W>0 &&  W<=min_ZYt
    const float m11 = !wz & !wle;         // W>0 &&  W> min_ZYt

    // ---------- 混合得到最终结果 ----------
    Y_tp1 = m00 * Y00 + m01 * Y01 + m10 * Y10 + m11 * Y11 * (X != 0); //哼，Huifang 改的（傲娇）！！！！！
    Z_tp1 = m00 * Z00 + m01 * Z01 + m10 * Z10 + m11 * Z11 * (X != 0); //哼，Huifang 改的（傲娇）！！！！！

        // P_tau_tp1 = d_P_tau[0] # 这个是P(tau=t+1)时刻的值
        // P_tau_gep_tp1 = d_P_tau[1] # 这个是P(tau>=t+1)时刻的值


    float P_tau_tp1 = 1 - P_tau_gep_tp1;

    curandStatePhilox4_32_10_t s = rng_states[idx];

    const float exp_term = expf((d_MU - l - 0.5f * d_SIGMA * d_SIGMA) * d_DELTA_T);
    const float sqrt_delta_t = sqrtf(d_DELTA_T);
    const float discount_factor = expf(-d_R * d_DELTA_T);

    // 生成随机数
    float random = curand_normal(&s);
    
    // d_temp += 1000 * random;

    // 计算 X(t+1)
    float X_tp1 = XmW * exp_term * expf(d_SIGMA * sqrt_delta_t * random);
    X_tp1 = fminf(X_tp1, d_MAX_X);
    
    // 查找值函数
    float V_tp1 = lookup_V(X_tp1, Y_tp1, Z_tp1, E_tp1); 

    float d_temp = discount_factor * (P_tau_tp1 * fmaxf(X_tp1, Y_tp1) + P_tau_gep_tp1 * V_tp1);

    atomicAdd(&d_d_results[idx], d_temp);

    if (thread_idx == 0) {
        // 优化代码
        // ─── 仅用 3 条浮点指令 + 1 条乘 fWt *= (t != 0) ──────────
        float fWt = W - d_A1 * fmaxf(W - min_ZYt, 0.0f);   // ← 已同时覆盖两种情况
        fWt       *= (t != 0);                           // t==0 → 置 0

        d_d_results[idx] = d_d_results[idx] / 1024 + fWt;
    }
    
}


// XYZEW kernel 实现
__global__ void XYZEW_kernel(int offset, int t, curandStatePhilox4_32_10_t *rng_states, float l, float a3, float P_tau_gep_tp1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx >= d_sXYZEW) return;

    // 计算索引
    int index_x = idx / d_sYZEW;
    int remainder = idx % d_sYZEW;
    int index_y = remainder / d_sZEW;
    remainder = remainder % d_sZEW;
    int index_z = remainder / d_sEW;
    remainder = remainder % d_sEW;
    int index_e = remainder / d_sW;
    int index_w = remainder % d_sW;

    // 获取值
    float X = d_d_X[index_x];
    float Y = d_d_Y[index_y];
    float Z = d_d_Z[index_z];
    int E = d_d_E[index_e];
    float W = d_d_W[index_w];

    if (W > Y) return;

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
    const float Y00 = fmaxf((1.0f + d_A2) * Y, XmW);          // W==0 && E==0
    const float Z00 = (1.0f + d_A2) * fmaxf(a3 * X,   Z);

    const float Y01 = fmaxf(Y, XmW);          // W==0 && E>0
    const float Z01 = fmaxf(Z, a3 * XmW);

    const float Y10 = fmaxf(Y - W, XmW);                     // W>0 && W<=min_ZYt
    const float Z10 = fmaxf(Z, a3 * XmW);

    const float t111  = fminf(Y - W,   Y * invX * XmW);            // W>0 && W>min_ZYt
    const float Y11 = fmaxf(t111, XmW);
    const float Z11 = fmaxf(Z * invX * XmW, a3 * XmW);

    // ---------- 4 个掩码 ----------
    const float m00 =  wz &  ez;          // W==0 &&  E==0
    const float m01 =  wz & !ez;          // W==0 &&  E>0
    const float m10 = !wz &  wle;         // W>0 &&  W<=min_ZYt
    const float m11 = !wz & !wle;         // W>0 &&  W> min_ZYt

    // ---------- 混合得到最终结果 ----------
    Y_tp1 = m00 * Y00 + m01 * Y01 + m10 * Y10 + m11 * Y11 * (X != 0); 
    Y_tp1 = fminf(Y_tp1, d_MAX_Y);
    Z_tp1 = m00 * Z00 + m01 * Z01 + m10 * Z10 + m11 * Z11 * (X != 0); 
    Z_tp1 = fminf(Z_tp1, d_MAX_Z);



    float P_tau_tp1 = 1 - P_tau_gep_tp1;
    // //Monte Carlo 模拟
    float d_temp = monte_carlo_simulation(
        XmW, Y_tp1, Z_tp1, E_tp1,
        P_tau_tp1, P_tau_gep_tp1,
        l, rng_states, idx
    );

    // 优化代码
    // ─── 仅用 3 条浮点指令 + 1 条乘 fWt *= (t != 0) ──────────
    float fWt = W - d_A1 * fmaxf(W - min_ZYt, 0.0f);   // ← 已同时覆盖两种情况
    fWt       *= (t != 0);                           // t==0 → 置 0

    
    // 存储结果
    d_d_results[idx] = d_temp / d_MOTECALO_NUMS + fWt;
    // d_results[idx] = d_temp;
}

// V_tp1 kernel 实现
__global__ void V_tp1_kernel(int offset, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx >= d_sXYZE) return;

    // 计算索引
    int index_x = idx / d_sYZE;
    int remainder = idx % d_sYZE;
    int index_y = remainder / d_sZE;
    remainder = remainder % d_sZE;
    int index_z = remainder / d_sE;
    // int index_e = remainder % sE;

    float X = d_d_X[index_x];
    float Y = d_d_Y[index_y];
    float Z = d_d_Z[index_z];
    // int E = d_d_E[index_e];

    int W_index = idx * d_SIZE_W;
    float max_w = d_d_results[W_index];

    if (t == 0) {
        d_d_V_tp1[idx] = max_w;//对应着d_results[index_x, index_y, index_z, index_e, 0]
        return;
    }

    // 查找最大值
    for (int i = 0; i < d_SIZE_W; i++) {

        if (Y >= d_d_W[i]) {
            float current = d_d_results[W_index + i];
            if (current > max_w) {
                max_w = current;
            }
        }
    }

    d_d_V_tp1[idx] = fmaxf(fmaxf(Y - d_A1 * (Y - fminf(Z, Y)), X), max_w);
    // d_d_V_tp1[idx] = fmaxf(fmaxf(Y - d_A1 * fmaxf((Y - fminf(Z, Y)), 0.0f), X), max_w) * (X != 0);
}


