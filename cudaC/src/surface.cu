#include <iostream>
#include <cuda_runtime.h>

// ... (CHECK宏) ...
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
/**
 * @brief Kernel to update a 3D cudaArray via a surface object.
 *
 * @param surfObj The surface object to write to.
 * @param width Width of the array.
 * @param height Height of the array.
 * @param depth Depth of the array.
 */
__global__ void update_via_surface_kernel(cudaSurfaceObject_t surfObj, int width, int height, int depth) {
    // 使用3D线程索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) {
        return;
    }

    // 计算新值
    float newValue = 777.0f;

    // --- 使用surf3Dwrite写入数据 ---
    // 注意：x坐标必须是字节偏移量！这是一个常见错误点。
    int x_bytes = x * sizeof(float);
    surf3Dwrite(newValue, surfObj, x_bytes, y, z);
}

void update_from_device(cudaArray_t& cuArray, int width, int height, int depth) {
    std::cout << "\n--- 方法二：从设备端Kernel更新 cudaArray (使用Surface) ---\n";

    // === 1. 创建可用于Surface的cudaArray ===
    // 如果已有的cuArray不是这样创建的，需要重新创建
    if (cuArray) CHECK(cudaFreeArray(cuArray));
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(width, height, depth);
    //  <<<<< 关键步骤：添加 cudaArraySurfaceLoadStore 标志 >>>>>
    CHECK(cudaMalloc3DArray(&cuArray, &channelDesc, extent, cudaArraySurfaceLoadStore));

    // === 2. 创建Surface对象 ===
    cudaSurfaceObject_t surfObj = 0;
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    CHECK(cudaCreateSurfaceObject(&surfObj, &resDesc));

    // === 3. 启动更新Kernel ===
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (depth + threadsPerBlock.z - 1) / threadsPerBlock.z);

    update_via_surface_kernel<<<numBlocks, threadsPerBlock>>>(surfObj, width, height, depth);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize()); // 等待kernel执行完毕

    std::cout << "cudaArray 已通过 Surface Kernel 更新完毕。\n";

    // === 4. 清理Surface对象 ===
    CHECK(cudaDestroySurfaceObject(surfObj));
}