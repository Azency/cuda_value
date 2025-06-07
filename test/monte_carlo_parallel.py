# 1. 首先修改kernel函数，添加W作为参数
@cuda.jit
def motecalo_kernel(d_result, d_random01, d_P_tau, d_V, d_args, d_W):
    idx = cuda.grid(1)
    if idx < d_result.size:
        randmon = d_random01[idx]
        # 获取当前W值
        W = d_W[idx // Motecalo_nums]  # 每个W值对应Motecalo_nums个随机数
        
        # 解包参数
        X = d_args[0]
        Y = d_args[1]
        Z = d_args[2]
        E = d_args[3]
        r = d_args[4]
        delta_t = d_args[5]
        mu = d_args[6]
        sigma = d_args[7]
        l = d_args[8]
        a2 = d_args[9]
        a3 = d_args[10]

        # 计算X_tp1
        X_tp1 = (X - W) * math.exp((mu - l - sigma ** 2 / 2) * delta_t + 
                                  sigma * math.sqrt(delta_t) * randmon)

        E_tp1 = E + W
        
        min_ZYt = min(Z, Y)
        if W == 0:
            if E == 0:
                Y_tp1 = (1 + a2) * max(X, Y)
                Z_tp1 = (1 + a2) * max(a3*X, Y)
            elif E > 0:
                Y_tp1 = max(X, Y)
                Z_tp1 = max(a3*X, Y)
        elif W > 0:
            if W <= min_ZYt:
                Y_tp1 = max(X - W, Y - W)
                Z_tp1 = max(a3*(X - W), Z)
            elif W > min_ZYt:
                Y_tp1 = max(X - W, min(Y - W, Y / X * (X - W)))
                Z_tp1 = max(a3*(X - W), Z / X * (X - W))
        
        V_tp1 = __lookup_V__(d_V, X_tp1, Y_tp1, Z_tp1, E_tp1)
        
        P_tau_tp1 = d_P_tau[0]
        P_tau_gep_tp1 = d_P_tau[1]
        
        d_result[idx] = math.exp(-r * delta_t) * (P_tau_tp1 * max(X_tp1, Y_tp1) + 
                                                 P_tau_gep_tp1 * V_tp1)

# 2. 修改kernel_W函数，使其处理多个W值
def kernel_W_parallel(kernel_args, all_W, Motecalo_nums, P_tau_t):
    try:
        # 将参数转移到GPU
        d_args = cuda.to_device(np.array(kernel_args, dtype=np.float64))
        d_W = cuda.to_device(all_W)
        d_P_tau = cuda.to_device(np.array(P_tau_t, dtype=np.float64))
        
        # 为每个W值生成随机数
        total_samples = len(all_W) * Motecalo_nums
        rng = np.random.default_rng()
        d_random01 = cuda.to_device(rng.normal(0, 1, total_samples))
        
        # 创建结果数组
        d_result = cuda.to_device(np.zeros(total_samples, dtype=np.float64))
        
        # 配置kernel参数
        threads_per_block = 1024
        blocks_per_grid = (total_samples + threads_per_block - 1) // threads_per_block
        
        # 运行kernel
        motecalo_kernel[blocks_per_grid, threads_per_block](
            d_result, d_random01, d_P_tau, d_V, d_args, d_W)
        
        # 计算每个W值的平均值
        d_means = cuda.to_device(np.zeros(len(all_W), dtype=np.float64))
        
        @cuda.jit
        def reduce_kernel(d_result, d_means, Motecalo_nums):
            idx = cuda.grid(1)
            if idx < d_means.size:
                start_idx = idx * Motecalo_nums
                end_idx = start_idx + Motecalo_nums
                sum_val = 0.0
                for i in range(start_idx, end_idx):
                    sum_val += d_result[i]
                d_means[idx] = sum_val / Motecalo_nums
        
        # 运行reduce kernel
        reduce_kernel[blocks_per_grid, threads_per_block](
            d_means, d_result, Motecalo_nums)
        
        # 获取结果
        means = d_means.copy_to_host()
        
        # 清理资源
        del d_args
        del d_W
        del d_P_tau
        del d_random01
        del d_result
        del d_means
        cuda.synchronize()
        
        return means
        
    except Exception as e:
        print(f"Error in kernel_W_parallel: {e}")
        cuda.close()
        cuda.reset()
        raise e

# 3. 主程序使用示例
def main():
    # 设置参数
    kernel_args = [X, Y, Z, E, r, delta_t, mu, sigma, l, a2, a3]
    all_W = np.linspace(0, 1, 10000, dtype=np.float64)
    Motecalo_nums = 50000
    P_tau = [0.9, 0.95]
    
    # 并行计算所有W值
    V_of_W = kernel_W_parallel(kernel_args, all_W, Motecalo_nums, P_tau)
    
    # 找到最大值
    max_V_of_W = np.max(V_of_W)
    print(f"Maximum value: {max_V_of_W}")
    
    return V_of_W, max_V_of_W