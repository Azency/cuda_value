// W 维度优化实现示例代码
// 方案：合并 kernel + warp-level reduction

// 优化后的 kernel：直接计算最大值，不存储所有中间结果
__global__ void WEYZX_kernel_optimized(
    int offset, 
    int t, 
    curandStatePhilox4_32_10_t *rng_states, 
    float l, 
    float a3, 
    float P_tau_gep_tp1,
    float *d_results_max  // 只存储最大值，维度 (E,Y,Z,X)
) {
    // 计算当前线程处理的 (E,Y,Z,X) 索引
    int idx_eyzx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx_eyzx >= d_sEYZX) return;

    int index_e = idx_eyzx / d_sYZX;
    int remainder = idx_eyzx % d_sYZX;
    int index_y = remainder / d_sZX;
    remainder = remainder % d_sZX;
    int index_z = remainder / d_sX;
    int index_x = remainder % d_sX;

    // 获取固定值
    float X = d_d_X[index_x];
    float Y = d_d_Y[index_y];
    float Z = d_d_Z[index_z];
    int E = d_d_E[index_e];

    // 使用共享内存存储每个 W 值的结果，然后进行 reduction
    __shared__ float s_results[256];  // 假设 blockDim.x <= 256
    __shared__ float s_max_result[1];

    float max_value = -FLT_MAX;
    int valid_w_count = 0;

    // 每个线程处理一个或多个 W 值
    // 方案A：每个线程处理一个 W 值（需要 size_W <= blockDim.x）
    int thread_w = threadIdx.x;
    if (thread_w < d_SIZE_W) {
        float W = d_d_W[thread_w];
        
        if (W <= Y) {  // 只处理有效的 W 值
            // 计算该 W 值对应的结果（与原 kernel 相同的逻辑）
            float min_ZYt = fminf(Z, Y);
            int E_tp1 = 1 * (E + W == 0);
            
            const float invX = __frcp_rn(X);
            const float XmW = fmaxf(X - W, 0.0f);
            const bool wz = (W == 0);
            const bool ez = (E_tp1 == 0);
            const bool wle = (W <= min_ZYt);

            // 计算 Y_tp1, Z_tp1（与原代码相同）
            const float Y00 = fmaxf((1.0f + d_A2) * Y, XmW);
            const float Y01 = fmaxf(Y, XmW);
            const float Y10 = fmaxf(Y - W, XmW);
            const float t111 = fminf(Y - W, Y * invX * XmW);
            const float Y11 = fmaxf(t111, XmW);

            float Y_tp1 = (wz & ez) * Y00 + (wz & !ez) * Y01 + 
                         (!wz & wle) * Y10 + (!wz & !wle) * Y11 * (X != 0);
            Y_tp1 = fmaxf(fminf(Y_tp1, d_MAX_Y), d_MIN_Y);

            float Z_tp1;  // 类似计算 Z_tp1
            // ... Z_tp1 计算代码 ...

            // Monte Carlo 模拟
            float P_tau_tp1 = 1 - P_tau_gep_tp1;
            
            // 注意：需要为每个 W 值生成独立的随机数
            // 这里需要调整随机数生成策略
            int rng_idx = idx_eyzx * d_SIZE_W + thread_w;
            float d_temp = monte_carlo_simulation(
                XmW, Y_tp1, Z_tp1, E_tp1,
                P_tau_tp1, P_tau_gep_tp1,
                l, rng_states, rng_idx
            );

            float fWt = W - d_A1 * fmaxf(W - min_ZYt, 0.0f);
            fWt *= (t != 0);

            float result = d_temp / d_MOTECALO_NUMS + fWt;
            s_results[threadIdx.x] = result;
            
            // 更新最大值
            if (result > max_value) {
                max_value = result;
            }
        } else {
            s_results[threadIdx.x] = -FLT_MAX;  // 无效值
        }
    } else {
        s_results[threadIdx.x] = -FLT_MAX;
    }

    __syncthreads();

    // Warp-level reduction 找最大值
    // 使用 shuffle 指令进行高效的 reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, max_value, offset);
        if (other > max_value) {
            max_value = other;
        }
    }

    // 第一个 warp 的第一个线程负责写入最终结果
    if (threadIdx.x % warpSize == 0) {
        // 还需要跨 warp reduction（如果 block 内有多个 warp）
        float warp_max = max_value;
        if (threadIdx.x < warpSize) {
            // 使用共享内存进行跨 warp reduction
            s_results[threadIdx.x / warpSize] = warp_max;
        }
    }

    __syncthreads();

    // Block-level reduction（如果需要）
    if (blockDim.x > warpSize) {
        if (threadIdx.x < blockDim.x / warpSize) {
            float local_max = s_results[threadIdx.x];
            for (int i = blockDim.x / warpSize; i < blockDim.x; i += warpSize) {
                if (i + threadIdx.x < blockDim.x) {
                    float other = s_results[i + threadIdx.x];
                    if (other > local_max) {
                        local_max = other;
                    }
                }
            }
            s_results[threadIdx.x] = local_max;
        }
        __syncthreads();
        
        // 最终 reduction
        if (threadIdx.x == 0) {
            float final_max = s_results[0];
            for (int i = 1; i < (blockDim.x + warpSize - 1) / warpSize; i++) {
                if (s_results[i] > final_max) {
                    final_max = s_results[i];
                }
            }
            d_results_max[idx_eyzx] = final_max;
        }
    } else {
        // 只有一个 warp，直接写入
        if (threadIdx.x == 0) {
            d_results_max[idx_eyzx] = max_value;
        }
    }
}

// 简化版本：如果 size_W <= 32，可以使用更简单的实现
__global__ void WEYZX_kernel_simple(
    int offset,
    int t,
    curandStatePhilox4_32_10_t *rng_states,
    float l,
    float a3,
    float P_tau_gep_tp1,
    float *d_results_max
) {
    int idx_eyzx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx_eyzx >= d_sEYZX) return;

    // ... 索引计算代码 ...

    float max_value = -FLT_MAX;

    // 如果 size_W <= 32，可以在一个 warp 内处理
    for (int i = 0; i < d_SIZE_W; i++) {
        float W = d_d_W[i];
        if (W > Y) continue;  // 跳过无效的 W

        // 计算该 W 值的结果
        // ... 计算逻辑 ...
        
        float result = /* 计算结果 */;
        if (result > max_value) {
            max_value = result;
        }
    }

    d_results_max[idx_eyzx] = max_value;
}

/*
 * 实现注意事项：
 * 
 * 1. **随机数生成**：
 *    - 原实现中每个 (W,E,Y,Z,X) 组合使用 idx 作为随机数种子
 *    - 优化后需要确保每个 W 值都有独立的随机数流
 *    - 可以使用 idx_eyzx * d_SIZE_W + w_index 作为随机数索引
 * 
 * 2. **内存占用**：
 *    - 原：d_results[sWEYZX] = size_W * size_E * size_Y * size_Z * size_X
 *    - 新：d_results_max[sEYZX] = size_E * size_Y * size_Z * size_X
 *    - 减少比例：1 - 1/size_W ≈ 95% (当 size_W=21)
 * 
 * 3. **性能考虑**：
 *    - 如果 size_W 较小（<=32），可以使用简单的循环
 *    - 如果 size_W 较大，需要使用 warp-level reduction
 *    - 需要考虑 W <= Y 的条件过滤
 * 
 * 4. **V_tp1_kernel 简化**：
 *    - 优化后不再需要 V_tp1_kernel，因为最大值已经在 WEYZX_kernel 中计算
 *    - 或者 V_tp1_kernel 只需要简单的处理：d_V_tp1 = max(d_results_max, ...)
 */

