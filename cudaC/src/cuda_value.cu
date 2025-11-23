#include "cuda_value.h"
// #include "config.h"

float h_MIN_X, h_MIN_Y, h_MIN_Z, h_MIN_W;
float h_MAX_X, h_MAX_Y, h_MAX_Z, h_MAX_W;
int h_SIZE_X, h_SIZE_Y, h_SIZE_Z, h_SIZE_E, h_SIZE_W;
int h_sWEYZX, h_sEYZX, h_sYZX, h_sZX, h_sX;

float h_SCALE_TO_INT_X, h_SCALE_TO_INT_Y, h_SCALE_TO_INT_Z;

__constant__ float d_MIN_X, d_MIN_Y, d_MIN_Z, d_MIN_W;
__constant__ float d_MAX_X, d_MAX_Y, d_MAX_Z, d_MAX_W;
__constant__ int d_SIZE_X, d_SIZE_Y, d_SIZE_Z, d_SIZE_E, d_SIZE_W;
__constant__ int d_sWEYZX, d_sEYZX, d_sYZX, d_sZX, d_sX;

__constant__ float d_SCALE_TO_INT_X, d_SCALE_TO_INT_Y, d_SCALE_TO_INT_Z;


float h_A1,h_P, h_INITIAL_INVESTMENT, h_DELTA_T;
__constant__ float d_A1, d_A2, d_R, d_MU, d_SIGMA, d_P, d_INITIAL_INVESTMENT, d_DELTA_T;
__constant__ int d_MOTECALO_NUMS;

float *d_X, *d_Y, *d_Z, *d_W, *d_V, *d_V_tp1, *d_results;
int *d_E;

__constant__ float *d_d_X, *d_d_Y, *d_d_Z, *d_d_W, *d_d_V, *d_d_V_tp1, *d_d_results;
__constant__ int *d_d_E;

// 2. === 使用 cudaArray 分配设备内存 ===
cudaArray_t cuArray0;
cudaArray_t cuArray1;
// 描述我们数据的格式（单通道32位浮点数）
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

cudaTextureObject_t texObj0 = 0;
cudaTextureObject_t texObj1 = 0;
cudaSurfaceObject_t surfObj0 = 0;
cudaSurfaceObject_t surfObj1 = 0;

__constant__ cudaTextureObject_t d_texObj0 = 0;
__constant__ cudaTextureObject_t d_texObj1 = 0;
__constant__ cudaSurfaceObject_t d_surfObj0 = 0;
__constant__ cudaSurfaceObject_t d_surfObj1 = 0;

// 随机数生成器（改为内核内即时初始化本地状态，取消全局大数组）




// 兼容全范围浮点数（含负数）的原子最大值更新
__device__ inline float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old_i = *address_as_i;
    float old_f = __int_as_float(old_i);
    while (old_f < val) {
        int assumed = old_i;
        old_i = atomicCAS(address_as_i, assumed, __float_as_int(val));
        if (old_i == assumed) break;
        old_f = __int_as_float(old_i);
    }
    return old_f;
}

// 将 d_results 初始化为 -INFINITY，避免 NaN 传播并便于取最大值
__global__ void init_results_neg_inf(float* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = -INFINITY;
    }
}

// 生成随机数
__global__ void setup(curandStatePhilox4_32_10_t *state, unsigned long seed, int PATHS)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= PATHS) return;
    /* sequence=tid, offset=0 → 线程独立子流 */
    curand_init(seed, tid, 0, &state[tid]);
}


__device__ int IDX_V(int e, int y, int z, int x) {
    // if(x*sYZEW + y*sZEW + z*sEW + e*sW + e >= SIZE_X * SIZE_Y * SIZE_Z * SIZE_E) {
    //     printf("Error: Index out of bounds\n");
    //     exit(1);
    // }
    int res = e*d_sYZX + y*d_sZX + z*d_sX + x;
    return res;
}

__host__ int h_IDX_V(int e, int y, int z, int x){
    int res = e*h_sYZX + y*h_sZX + z*h_sX + x;
    return res;
}


// 检查并恢复CUDA设备
static void check_and_recover_device() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "ERROR: No CUDA-capable device detected. Error: %s\n", 
                cudaGetErrorString(err));
        // 尝试重置设备
        cudaDeviceReset();
        // 再次检查
        err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            fprintf(stderr, "FATAL: Cannot recover CUDA device. Exiting.\n");
            exit(EXIT_FAILURE);
        }
        fprintf(stderr, "WARNING: CUDA device recovered after reset.\n");
    }
    
    // 确保设备已设置
    int currentDevice = -1;
    cudaGetDevice(&currentDevice);
    if (currentDevice < 0) {
        cudaSetDevice(0);
    }
}

void init_global_config(
    float min_X, float max_X, int size_X,
    float min_Y, float max_Y, int size_Y,
    float min_Z, float max_Z, int size_Z,
    int min_E, int max_E, int size_E,
    float min_W, float max_W, int size_W,
    float a1, float a2, float r, float mu, float sigma, int motecalo_nums, float p, float initial_investment
){
    // 在初始化前检查设备
    check_and_recover_device();
    
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

    h_sWEYZX = size_W * size_E * size_Y * size_Z * size_X;
    h_sEYZX = size_E * size_Y * size_Z * size_X;
    h_sYZX = size_Y * size_Z * size_X;
    h_sZX = size_Z * size_X;
    h_sX = size_X;


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


    cudaMemcpyToSymbolAsync(d_sWEYZX, &h_sWEYZX, sizeof(int));
    cudaMemcpyToSymbolAsync(d_sEYZX, &h_sEYZX, sizeof(int));
    cudaMemcpyToSymbolAsync(d_sYZX, &h_sYZX, sizeof(int));
    cudaMemcpyToSymbolAsync(d_sZX, &h_sZX, sizeof(int));
    cudaMemcpyToSymbolAsync(d_sX, &h_sX, sizeof(int));

    cudaMemcpyToSymbolAsync(d_SCALE_TO_INT_X, &h_SCALE_TO_INT_X, sizeof(float));
    cudaMemcpyToSymbolAsync(d_SCALE_TO_INT_Y, &h_SCALE_TO_INT_Y, sizeof(float));
    cudaMemcpyToSymbolAsync(d_SCALE_TO_INT_Z, &h_SCALE_TO_INT_Z, sizeof(float));


 

}

void init_global_XYZEW_V() {
    // 在初始化前检查设备
    check_and_recover_device();
    
   // 初始化XYZEW_V
    float *h_X = (float *)malloc(h_SIZE_X * sizeof(float));
    float *h_Y = (float *)malloc(h_SIZE_Y * sizeof(float));
    float *h_Z = (float *)malloc(h_SIZE_Z * sizeof(float));
    int   *h_E = (int   *)malloc(h_SIZE_E * sizeof(int));
    float *h_W = (float *)malloc(h_SIZE_W * sizeof(float));
    float *h_V = (float *)malloc(h_sEYZX * sizeof(float));
    
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
    for (int x = 0; x < h_SIZE_X; x++) {
        for (int y = 0; y < h_SIZE_Y; y++) {
            for (int z = 0; z < h_SIZE_Z; z++) {
                float min_ZY = fminf(h_Z[z], h_Y[y]);
                float term = (h_Y[y] <= min_ZY) ? 
                            h_Y[y] : 
                            h_Y[y] - h_A1 * (h_Y[y] - min_ZY);
                float result = fmaxf(h_X[x], term);
                
                // 对 E 的两个维度都赋值
                h_V[h_IDX_V(0, y, z, x)] = result;
                h_V[h_IDX_V(1, y, z, x)] = result;

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
    cudaMalloc(&d_V, h_sEYZX * sizeof(float));
    cudaMalloc(&d_V_tp1, h_sEYZX * sizeof(float));
    // 优化：d_results 只存储 (E,Y,Z,X) 的最大值，不再包含 W 维度
    cudaMalloc(&d_results, h_sEYZX * sizeof(float));
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
    cudaMemcpy(d_V, h_V, h_sEYZX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_tp1, h_V, h_sEYZX * sizeof(float), cudaMemcpyHostToDevice);
    

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

    init_texture_surface_object();
    printf("init_texture_surface_object is done\n");

    // 释放主机内存
    free(h_X);
    free(h_Y);
    free(h_Z);
    free(h_E);
    free(h_W);
    free(h_V);

    // 随机数生成器：已改为内核内本地初始化，无需全局分配
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
    if (cuArray0) cudaFreeArray(cuArray0);
    if (cuArray1) cudaFreeArray(cuArray1);

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

void init_random_state() { /* 已改为线程本地初始化，保留空实现以兼容调用方 */ }

void init_texture_surface_object() {
    
    cudaExtent extent = make_cudaExtent(h_SIZE_X, h_SIZE_Z, h_SIZE_Y);
    cudaMalloc3DArray(&cuArray0, &channelDesc, extent, cudaArrayDefault);
    cudaMalloc3DArray(&cuArray1, &channelDesc, extent, cudaArrayDefault);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;

    copyParams.dstArray = cuArray0;
    copyParams.srcPtr = make_cudaPitchedPtr(d_V, (h_SIZE_X) * sizeof(float), h_SIZE_X, h_SIZE_Z);
    cudaMemcpy3D(&copyParams);
    copyParams.dstArray = cuArray1;
    copyParams.srcPtr = make_cudaPitchedPtr(d_V+h_sYZX, (h_SIZE_X) * sizeof(float), h_SIZE_X, h_SIZE_Z);
    cudaMemcpy3D(&copyParams);

    // -- 绑定纹理对象
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray; // 注意，类型变了！

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode   = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // -- 创建纹理对象
    resDesc.res.array.array = cuArray0;  
    cudaCreateTextureObject(&texObj0, &resDesc, &texDesc, NULL);
    cudaCreateSurfaceObject(&surfObj0, &resDesc);
    resDesc.res.array.array = cuArray1;
    cudaCreateTextureObject(&texObj1, &resDesc, &texDesc, NULL);
    cudaCreateSurfaceObject(&surfObj1, &resDesc);

    cudaMemcpyToSymbolAsync(d_texObj0, &texObj0, sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbolAsync(d_texObj1, &texObj1, sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbolAsync(d_surfObj0, &surfObj0, sizeof(cudaSurfaceObject_t));
    cudaMemcpyToSymbolAsync(d_surfObj1, &surfObj1, sizeof(cudaSurfaceObject_t));


}

void copy_cudaarray_to_vtp1() {
    cudaExtent extent = make_cudaExtent(h_SIZE_X, h_SIZE_Z, h_SIZE_Y);
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;

    copyParams.srcArray = cuArray0;
    copyParams.dstPtr = make_cudaPitchedPtr(d_V_tp1, (h_SIZE_X) * sizeof(float), h_SIZE_X, h_SIZE_Z);
    cudaMemcpy3D(&copyParams);
 
    copyParams.srcArray = cuArray1;
    copyParams.dstPtr = make_cudaPitchedPtr(d_V_tp1+h_sYZX, (h_SIZE_X) * sizeof(float), h_SIZE_X, h_SIZE_Z);
    cudaMemcpy3D(&copyParams);

}



// 一轮计算后重置Vtp1
void reset_Vtp1() {
    // cudaMemcpy(d_V_tp1, d_V, h_sEYZX * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaExtent extent = make_cudaExtent(h_SIZE_X, h_SIZE_Z, h_SIZE_Y);
    cudaMemcpy3DParms copyParams = {0};
    copyParams.dstArray = cuArray0;
    copyParams.srcPtr = make_cudaPitchedPtr(d_V, (h_SIZE_X) * sizeof(float), h_SIZE_X, h_SIZE_Z);
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);

    copyParams.dstArray = cuArray1;
    copyParams.srcPtr = make_cudaPitchedPtr(d_V+h_sYZX, (h_SIZE_X) * sizeof(float), h_SIZE_X, h_SIZE_Z);
    cudaMemcpy3D(&copyParams);

    printf("reset_Vtp1 is done\n");
}




// 查表函数
__device__ float lookup_V(float X, float Y, float Z, int E) {
    // int E_int = E;
    // int X_down = (int)floorf((X - d_MIN_X) * d_SCALE_TO_INT_X);
    // int Y_down = (int)floorf((Y - d_MIN_Y) * d_SCALE_TO_INT_Y);
    // int Z_down = (int)floorf((Z - d_MIN_Z) * d_SCALE_TO_INT_Z);
    // int X_up   = fminf(X_down + 1, d_SIZE_X - 1);
    // int Y_up   = fminf(Y_down + 1, d_SIZE_Y - 1);
    // int Z_up   = fminf(Z_down + 1, d_SIZE_Z - 1);


    // float dx = (X - d_MIN_X) * d_SCALE_TO_INT_X - X_down;
    // float dy = (Y - d_MIN_Y) * d_SCALE_TO_INT_Y - Y_down;
    // float dz = (Z - d_MIN_Z) * d_SCALE_TO_INT_Z - Z_down;

    // float res = (1 - dx) * (1 - dy) * (1 - dz) * d_d_V_tp1[IDX_V(E_int, Y_down, Z_down, X_down)] + 
    //             dx * (1 - dy) * (1 - dz) * d_d_V_tp1[IDX_V(E_int, Y_down, Z_down, X_up)] + 
    //             (1 - dx) * dy * (1 - dz) * d_d_V_tp1[IDX_V(E_int, Y_up, Z_down, X_down)] + 
    //             dx * dy * (1 - dz) * d_d_V_tp1[IDX_V(E_int, Y_up, Z_down, X_up)] + 
    //             (1 - dx) * (1 - dy) * dz * d_d_V_tp1[IDX_V(E_int, Y_down, Z_up, X_down)] + 
    //             dx * (1 - dy) * dz * d_d_V_tp1[IDX_V(E_int, Y_down, Z_up, X_up)] + 
    //             (1 - dx) * dy * dz * d_d_V_tp1[IDX_V(E_int, Y_up, Z_up, X_down)] + 
    //             dx * dy * dz * d_d_V_tp1[IDX_V(E_int, Y_up, Z_up, X_up)];

    float X1 = (X - d_MIN_X) * d_SCALE_TO_INT_X + 0.5f;
    float Y1 = (Y - d_MIN_Y) * d_SCALE_TO_INT_Y + 0.5f;
    float Z1 = (Z - d_MIN_Z) * d_SCALE_TO_INT_Z + 0.5f;
    float res = 0;
    if (E == 0) {
        res = tex3D<float>(d_texObj0, X1, Z1, Y1);
    } else {
        res = tex3D<float>(d_texObj1, X1, Z1, Y1);
    }
    return res;              

}



// 设备函数实现
__device__ float monte_carlo_simulation(float XmW, float Y_tp1, float Z_tp1, int E_tp1, float P_tau_tp1, float P_tau_gep_tp1, float l, curandStatePhilox4_32_10_t * state) {
    float d_temp = 0.0f;
    curandStatePhilox4_32_10_t s = *state;
    
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

    *state = s;
    
    return d_temp ;
}



// XYZEW kernel 实现（优化版本：直接计算最大值，不存储所有 W 值）
__global__ void WEYZX_kernel(int offset, int t, int T, float l, float a3, float P_tau_gep_tp1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx >= d_sEYZX) return;

    // 计算 (E,Y,Z,X) 索引
    int index_e = idx / d_sYZX;
    int remainder = idx % d_sYZX;
    int index_y = remainder / d_sZX;
    remainder = remainder % d_sZX;
    int index_z = remainder / d_sX;
    int index_x = remainder % d_sX;

    // 获取 (X,Y,Z,E)
    float X = d_d_X[index_x];
    float Y = d_d_Y[index_y];
    float Z = d_d_Z[index_z];
    int E = d_d_E[index_e];

    float min_ZYt_base = fminf(Z, Y);
    const float invX = (X != 0.0f) ? __frcp_rn(X) : 0.0f;
    float best = -INFINITY;
    bool has_valid_w = false;

    // 每线程本地 RNG 状态（跨 t 推进；w 维度作为额外偏移）
    for (int index_w = 0; index_w < d_SIZE_W; ++index_w) {
        float W = d_d_W[index_w];
        if (W > Y) continue;
        has_valid_w = true;

        float XmW = fmaxf(X - W, 0.0f);
        bool  wz  = (W == 0);
        int   E_tp1 = 1 * (E + W == 0);
        bool  ez  = (E_tp1 == 0);
        bool  wle = (W <= min_ZYt_base);

        const float Y00 = fmaxf((1.0f + d_A2) * Y, XmW);
        const float Z00 = a3 * fmaxf((1.0f + d_A2) * Y, XmW);
        const float Y01 = fmaxf(Y, XmW);
        const float Z01 = fmaxf(Z, a3 * XmW);
        const float Y10 = fmaxf(Y - W, XmW);
        const float Z10 = fmaxf(Z, a3 * XmW);
        const float t111 = fminf(Y - W, Y * invX * XmW);
        const float Y11 = fmaxf(t111, XmW);
        const float Z11 = fmaxf(Z * invX * XmW, a3 * XmW);

        float Y_tp1 = (wz & ez) * Y00 + (wz & !ez) * Y01 + (!wz & wle) * Y10 + (!wz & !wle) * Y11 * (X != 0);
        Y_tp1 = fmaxf(fminf(Y_tp1, d_MAX_Y), d_MIN_Y);
        float Z_tp1 = (wz & ez) * Z00 + (wz & !ez) * Z01 + (!wz & wle) * Z10 + (!wz & !wle) * Z11 * (X != 0);
        Z_tp1 = fmaxf(fminf(Z_tp1, d_MAX_Z), d_MIN_Z);

        float P_tau_tp1 = 1 - P_tau_gep_tp1;

        curandStatePhilox4_32_10_t s;
        // 与旧版一致：每个 (W,E,Y,Z,X) 用独立 subsequence，时间步用 offset 推进
        unsigned long long seq = (unsigned long long)index_w * (unsigned long long)d_sEYZX
                                + (unsigned long long)idx; // idx 是 (E,Y,Z,X) 线性索引
        unsigned long long ofs = (unsigned long long)(T - 1 - t) * (unsigned long long)d_MOTECALO_NUMS;
        curand_init(101ULL, seq, ofs, &s);

        float d_temp = monte_carlo_simulation(
            XmW, Y_tp1, Z_tp1, E_tp1,
            P_tau_tp1, P_tau_gep_tp1,
            l, &s
        );

        float fWt = W - d_A1 * fmaxf(W - min_ZYt_base, 0.0f);
        fWt *= (t != 0);
        float result = d_temp / d_MOTECALO_NUMS + fWt;
        best = fmaxf(best, result);
    }

    // 如果所有 W 值都被跳过，使用一个合理的默认值
    // 这应该很少发生，但为了安全起见，我们使用边界值
    if (!has_valid_w) {
        // 如果没有有效的 W 值，使用边界条件：max(X, Y - A1*(Y - min(Z,Y)))
        best = fmaxf(fmaxf(Y - d_A1 * (Y - min_ZYt_base), X), 0.0f);
    }

    // 写回 (E,Y,Z,X) 的最大值，后续由 V_tp1_kernel 合成边界再写 surface
    d_d_results[idx] = best;
}

// V_tp1 kernel 实现
__global__ void V_tp1_kernel(int offset, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx >= d_sEYZX) return;

    // 计算索引
    int index_e = idx / d_sYZX;
    int remainder = idx % d_sYZX;
    int index_y = remainder / d_sZX;
    remainder = remainder % d_sZX;
    int index_z = remainder / d_sX;
    int index_x = remainder % d_sX;

    float X = d_d_X[index_x];
    float Y = d_d_Y[index_y];
    float Z = d_d_Z[index_z];

    // 优化：d_results 已经包含最大值，直接读取
    float max_w = d_d_results[idx];

    if (t == 0) {
        // 优化：直接使用最大值
        if (index_e == 0) {
            surf3Dwrite(max_w, d_surfObj0, index_x * sizeof(float), index_z, index_y);
        } else if (index_e == 1) {
            surf3Dwrite(max_w, d_surfObj1, index_x * sizeof(float), index_z, index_y);
        } 
        return;
    }

    // 优化：d_results 已经包含最大值，不需要再遍历 W 值
    float temp = fmaxf(fmaxf(Y - d_A1 * (Y - fminf(Z, Y)), X), max_w);

    if (index_e == 0) {
        surf3Dwrite(temp, d_surfObj0, index_x * sizeof(float), index_z, index_y);
    } else if (index_e == 1) {
        surf3Dwrite(temp, d_surfObj1, index_x * sizeof(float), index_z, index_y);
    } 
    // d_d_V_tp1[idx] = fmaxf(fmaxf(Y - d_A1 * fmaxf((Y - fminf(Z, Y)), 0.0f), X), max_w) * (X != 0);
}

//used for test function 
__global__ void test_array_kernel(cudaTextureObject_t texObj0, cudaTextureObject_t texObj1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_sWEYZX) return;

    d_d_results[idx] = 0;

    // 计算索引
    int index_w = idx / d_sEYZX;
    int remainder = idx % d_sEYZX;
    int index_e = remainder / d_sYZX;
    remainder = remainder % d_sYZX;
    int index_y = remainder / d_sZX;
    remainder = remainder % d_sZX;
    int index_z = remainder / d_sX;
    int index_x = remainder % d_sX;

    float X = d_d_X[index_x];
    float Y = d_d_Y[index_y];
    float Z = d_d_Z[index_z];
    int   E = d_d_E[index_e];
    float W = d_d_W[index_w];

    if (W > 0) return;

    // X = d_MAX_X/2;
    // Y = d_MAX_Y/2;
    // Z = d_MAX_Z/2;


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

    // float dx   = 1.0f;
    // float dy   = 1.0f; 
    // float dz   = 1.0f;

    // float dx = (X - d_MIN_X) * d_SCALE_TO_INT_X - X_down;
    // float dy = (Y - d_MIN_Y) * d_SCALE_TO_INT_Y - Y_down;
    // float dz = (Z - d_MIN_Z) * d_SCALE_TO_INT_Z - Z_down;

    float res = (1 - dx) * (1 - dy) * (1 - dz) * d_d_V_tp1[IDX_V(E_int, Y_down, Z_down, X_down)] + 
                dx * (1 - dy) * (1 - dz) * d_d_V_tp1[IDX_V(E_int, Y_down, Z_down, X_up)] + 
                (1 - dx) * dy * (1 - dz) * d_d_V_tp1[IDX_V(E_int, Y_up, Z_down, X_down)] + 
                dx * dy * (1 - dz) * d_d_V_tp1[IDX_V(E_int, Y_up, Z_down, X_up)] + 
                (1 - dx) * (1 - dy) * dz * d_d_V_tp1[IDX_V(E_int, Y_down, Z_up, X_down)] + 
                dx * (1 - dy) * dz * d_d_V_tp1[IDX_V(E_int, Y_down, Z_up, X_up)] + 
                (1 - dx) * dy * dz * d_d_V_tp1[IDX_V(E_int, Y_up, Z_up, X_down)] + 
                dx * dy * dz * d_d_V_tp1[IDX_V(E_int, Y_up, Z_up, X_up)];


    // float X1 = fminf((X - d_MIN_X) * d_SCALE_TO_INT_X, d_SIZE_X - 1) + 0.5f;
    // float Y1 = fminf((Y - d_MIN_Y) * d_SCALE_TO_INT_Y, d_SIZE_Y - 1) + 0.5f;
    // float Z1 = fminf((Z - d_MIN_Z) * d_SCALE_TO_INT_Z, d_SIZE_Z - 1) + 0.5f;

    float X1 = index_x + 0.5f;
    float Y1 = index_y + 0.5f;
    float Z1 = index_z + 0.5f;

    float res2;
    if (E_int == 0) {
        res2 = tex3D<float>(texObj0, X1, Z1, Y1);
    } else {
        res2 = tex3D<float>(texObj1, X1, Z1, Y1);
    }

    d_d_results[idx] = res2 - res;

}




float compute_l(float l, float * trans_tau_d, int T) {
    // 在计算前检查设备
    check_and_recover_device();
    
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
    init_random_state();

    // 设置block和grid
    dim3 block(1024);
    dim3 grid((h_sEYZX + block.x - 1) / block.x);

    dim3 block2(1024);
    dim3 grid2((h_sEYZX + block2.x - 1) / block2.x);
    for (int t = T-1; t >= 0; t--) {
        float P_tau_t = trans_tau_d[t];
        
        // 初始化 d_results 为 -INFINITY，避免 NaN 传播，便于随后取最大值
        init_results_neg_inf<<<grid, block>>>(d_results, h_sEYZX);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // 同步以确保内核完成并捕获运行时错误
        
        // 计算 W 的最大值到 d_results
        WEYZX_kernel<<<grid, block>>>(0, t, T, l, a3, P_tau_t);
        CUDA_CHECK(cudaGetLastError());     // launch
        CUDA_CHECK(cudaDeviceSynchronize()); // 同步以确保内核完成并捕获运行时错误
        
        // 合成边界并写入 surface
        V_tp1_kernel<<<grid2, block2>>>(0, t);
        CUDA_CHECK(cudaGetLastError());     // launch
        CUDA_CHECK(cudaDeviceSynchronize()); // 同步以确保内核完成并捕获运行时错误


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


    return output;
}

