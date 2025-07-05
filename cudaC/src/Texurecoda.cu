#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// (CHECK宏定义保持不变)
#define CHECK(call)                                                            \
do {                                                                           \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                      \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)


// Kernel保持不变，它不关心底层是Array还是Linear Memory
__global__ void trilinear_interpolation_kernel(cudaTextureObject_t texObj,
                                               const float3* input_coords,
                                               float* output_results,
                                               int num_coords)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_coords) {
        return;
    }
    float3 coord = input_coords[idx];
    output_results[idx] = tex3D<float>(texObj, coord.x, coord.y, coord.z);
}


int main() {
    // 1. 定义Matrix和Host数据 (这部分不变)
    const int WIDTH = 4;
    const int HEIGHT = 4;
    const int DEPTH = 4;
    std::vector<float> h_data(WIDTH * HEIGHT * DEPTH);
    for (int z = 0; z < DEPTH; ++z) {
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                h_data[z * WIDTH * HEIGHT + y * WIDTH + x] = (float)(z * 1000 + y * 100 + x);
            }
        }
    }

    // 2. === 使用 cudaArray 分配设备内存 ===
    cudaArray_t cuArray;
    // 描述我们数据的格式（单通道32位浮点数）
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    //  <<<<< 这里就是答案！维度在这里被设置 >>>>>
    // 使用cudaMalloc3DArray来分配一个3D的cudaArray，并传入维度
    cudaExtent extent = make_cudaExtent(WIDTH, HEIGHT, DEPTH);
    CHECK(cudaMalloc3DArray(&cuArray, &channelDesc, extent, cudaArrayDefault));

    // 3. === 使用 cudaMemcpy3D 将数据拷贝到 cudaArray ===
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(h_data.data(), WIDTH * sizeof(float), WIDTH, HEIGHT);
    copyParams.dstArray = cuArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    CHECK(cudaMemcpy3D(&copyParams));

    // 4. === 创建并配置纹理对象，这次绑定到cudaArray ===
    cudaTextureObject_t texObj = 0;

    // -- 指定资源 (我们的cudaArray)
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray; // 注意，类型变了！
    resDesc.res.array.array = cuArray;       // 绑定到cuArray

    // -- 指定纹理参数 (这部分和之前一样)
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // -- 创建纹理对象
    CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    // 5. 准备坐标、启动内核、验证结果 (这部分和之前完全一样)
    // ... (代码省略，和上一个例子相同) ...
    std::vector<float3> h_coords;
    h_coords.push_back({1.5f, 1.0f, 1.0f}); // 预期: 1101.5
    h_coords.push_back({-10.0f, 1.0f, 1.0f}); // 预期: 1100
    // ...
    float3* d_coords = nullptr;
    float* d_results = nullptr;
    const int num_coords = h_coords.size();
    CHECK(cudaMalloc(&d_coords, num_coords * sizeof(float3)));
    CHECK(cudaMalloc(&d_results, num_coords * sizeof(float)));
    CHECK(cudaMemcpy(d_coords, h_coords.data(), num_coords * sizeof(float3), cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks_per_grid = (num_coords + threads_per_block - 1) / threads_per_block;
    trilinear_interpolation_kernel<<<blocks_per_grid, threads_per_block>>>(texObj, d_coords, d_results, num_coords);
    CHECK(cudaGetLastError());
    
    std::vector<float> h_results(num_coords);
    CHECK(cudaMemcpy(h_results.data(), d_results, num_coords * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\n--- 插值结果验证 (使用 cudaArray) ---\n";
    for (int i = 0; i < num_coords; ++i) {
        std::cout << "坐标 (" << h_coords[i].x << ", " << h_coords[i].y << ", " << h_coords[i].z
                  << ") 的插值结果是: " << h_results[i] << std::endl;
    }


    // 6. 清理资源
    CHECK(cudaDestroyTextureObject(texObj));
    CHECK(cudaFreeArray(cuArray)); // 注意，使用cudaFreeArray
    CHECK(cudaFree(d_coords));
    CHECK(cudaFree(d_results));

    return 0;
}