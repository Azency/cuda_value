from numba import cuda, prange
import numba
import numpy as np
from Prob import D_Prob
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32



a1 = 0.1
min_XYZ = 0
max_XYZ = 10
size_XYZ = 200
size_W = 100

# 将0-10的区间均匀分成X_size份
X = np.linspace(min_XYZ, max_XYZ, size_XYZ, dtype=np.float64)
Y = np.linspace(min_XYZ, max_XYZ, size_XYZ, dtype=np.float64)
Z = np.linspace(min_XYZ, max_XYZ, size_XYZ, dtype=np.float64)
E = np.arange(2, dtype=np.int16) 

W = np.linspace(0, 1, size_W, dtype=np.float64)

def init_global_V(X, Y, Z, E, a1):
    host_array = np.zeros((size_XYZ, size_XYZ, size_XYZ, 2), dtype=np.float64)
    # 创建网格
    X_mesh, Y_mesh, Z_mesh = np.meshgrid(X, Y, Z, indexing='ij')

    print(X_mesh.shape)
    
    # 计算min(Z, Y)
    min_ZY = np.minimum(Z_mesh, Y_mesh)
    
    # 计算Y - a1*(Y - min(Z, Y))
    term = Y_mesh - a1 * (Y_mesh - min_ZY)
    
    # 计算max(X, term)
    result = np.maximum(X_mesh, term)
    
    # 将结果赋值给host_array
    host_array[..., 0] = result
    host_array[..., 1] = result
    
    # 转移到GPU
    global d_V
    d_V = cuda.to_device(host_array)

init_global_V(X, Y, Z, E, a1)


scale_to_int = float(size_XYZ)/(max_XYZ-min_XYZ)

@cuda.jit(device=True)
def __lookup_V__(d_V, X, Y, Z, E):
    X_int = int(math.floor((X - min_XYZ) * scale_to_int))
    Y_int = int(math.floor((Y - min_XYZ) * scale_to_int))
    Z_int = int(math.floor((Z - min_XYZ) * scale_to_int))
    E_int = int(E)
    return d_V[X_int, Y_int, Z_int, E_int]

# ------------------------------    --------------------------------

# 创建kernel函数，每个线程计算一个motecalo值
@cuda.jit
def motecalo_kernel(d_result, d_random01, d_P_tau, d_V, d_args):
    idx = cuda.grid(1)
    randmon = d_random01[idx]
    # [X, Y, Z, E, W, r, delta_t, mu, sigma, l] = d_args
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
    W = d_args[11]

    X_tp1 = (X - W) * math.exp( (mu - l - sigma ** 2 / 2) * delta_t + sigma * math.sqrt(delta_t) * randmon)

    E_tp1 = E + W
    
    min_ZYt = min(Z, Y)
    if W == 0:
        if E == 0:
            Y_tp1 = (1 + a2) * max(X, Y)
            Z_tp1 = (1 + a2) * max(a3*X, Y)
        elif E > 0:
            Y_tp1 = max(X, Y)
            Z_tp1 = max(a3*X, Y)
    elif W >0 :
        if W <= min_ZYt:
            Y_tp1 = max(X - W , Y - W)
            Z_tp1 = max(a3*(X - W), Z)
        elif W > min_ZYt:
            Y_tp1 = max(X - W, 
                           min(Y - W, Y / X * (X - W))
                           )
            Z_tp1 = max(a3*(X - W), 
                            Z / X * (X - W)
                           )
            
    

    V_tp1 = __lookup_V__(d_V, X_tp1, Y_tp1, Z_tp1, E_tp1)
    
    P_tau_tp1 = d_P_tau[0] # 这个是P(tau=t+1)时刻的值
    P_tau_gep_tp1 = d_P_tau[1] # 这个是P(tau>=t+1)时刻的值
    
    d_result[idx] = math.exp(-r * delta_t) * (P_tau_tp1 * max(X_tp1, Y_tp1) + P_tau_gep_tp1 * V_tp1)


# 创建kernel函数，每个线程计算一个motecalo值
@cuda.jit
def motecalo_kernel2(d_result, rng_states, d_P_tau, d_V, d_args):
    idx = cuda.grid(1)

    randomn = xoroshiro128p_normal_float32(rng_states, idx)
    # [X, Y, Z, E, W, r, delta_t, mu, sigma, l] = d_args
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
    W = d_args[11]

    X_tp1 = (X - W) * math.exp( (mu - l - sigma ** 2 / 2) * delta_t + sigma * math.sqrt(delta_t) * randomn)

    E_tp1 = E + W
    
    min_ZYt = min(Z, Y)
    if W == 0:
        if E == 0:
            Y_tp1 = (1 + a2) * max(X, Y)
            Z_tp1 = (1 + a2) * max(a3*X, Y)
        elif E > 0:
            Y_tp1 = max(X, Y)
            Z_tp1 = max(a3*X, Y)
    elif W >0 :
        if W <= min_ZYt:
            Y_tp1 = max(X - W , Y - W)
            Z_tp1 = max(a3*(X - W), Z)
        elif W > min_ZYt:
            Y_tp1 = max(X - W, 
                           min(Y - W, Y / X * (X - W))
                           )
            Z_tp1 = max(a3*(X - W), 
                            Z / X * (X - W)
                           )
            
    

    V_tp1 = __lookup_V__(d_V, X_tp1, Y_tp1, Z_tp1, E_tp1)
    
    P_tau_tp1 = d_P_tau[0] # 这个是P(tau=t+1)时刻的值
    P_tau_gep_tp1 = d_P_tau[1] # 这个是P(tau>=t+1)时刻的值
    
    d_result[idx] = math.exp(-r * delta_t) * (P_tau_tp1 * max(X_tp1, Y_tp1) + P_tau_gep_tp1 * V_tp1)

@cuda.reduce
def sum_reduce(a, b):
    return a + b

def kernel_W(kernel_args, W, Motecalo_nums, P_tau_t):
    kernel_args.append(W)
    d_args = cuda.to_device(np.array(kernel_args, dtype=np.float64))


    # 使用numpy创建随机数组
    rng = np.random.default_rng()

    # 使用 rng.normal() 生成标准正态分布的样本
    d_random01 = cuda.to_device(rng.normal(0, 1, Motecalo_nums))

    d_result = cuda.to_device(np.zeros(Motecalo_nums, dtype=np.float64))

    # 将P(tau=t)和P(tau>=t)存入GPU
    d_P_tau = cuda.to_device(np.array(P_tau_t, dtype=np.float64))

    all_threads = Motecalo_nums
    # rng_states = create_xoroshiro128p_states(all_threads, seed=1)
    block_size = 1024
    blocks_per_grid = (all_threads + block_size - 1) // block_size
    motecalo_kernel[blocks_per_grid, block_size](d_result, d_random01, d_P_tau, d_V, d_args)

    cuda.synchronize()

    
    avg = sum_reduce(d_result)/Motecalo_nums

    del d_args
    del d_random01
    del d_result
    del d_P_tau

    cuda.synchronize()


    return avg


def kernel_W2(kernel_args, W, Motecalo_nums, P_tau_t):
    kernel_args.append(W)
    d_args = cuda.to_device(np.array(kernel_args, dtype=np.float64))


    # # 使用numpy创建随机数组
    # rng = np.random.default_rng()

    # # 使用 rng.normal() 生成标准正态分布的样本
    # d_random01 = cuda.to_device(rng.normal(0, 1, Motecalo_nums))

    # d_result = cuda.to_device(np.zeros(Motecalo_nums, dtype=np.float64))

    # 将P(tau=t)和P(tau>=t)存入GPU
    d_P_tau = cuda.to_device(np.array(P_tau_t, dtype=np.float64))

    all_threads = Motecalo_nums
    rng_states = create_xoroshiro128p_states(all_threads, seed=1)
    block_size = 1024
    blocks_per_grid = (all_threads + block_size - 1) // block_size
    motecalo_kernel2[blocks_per_grid, block_size](d_result, rng_states, d_P_tau, d_V, d_args)

    cuda.synchronize()

    
    avg = sum_reduce(d_result)/Motecalo_nums

    # del d_random01
    del d_result
    del d_P_tau

    cuda.synchronize()


    return avg


numba.njit(parallel=True, nogil=True)
def pthread_W(kernel_args, all_W, Motecalo_nums, P_tau_t):
    V_of_W = []
    for i in prange(len(all_W)):
        avg = kernel_W(kernel_args, W, Motecalo_nums, P_tau_t)
        V_of_W.append(avg)

    max_V_of_W = max(V_of_W)
    return max_V_of_W


if __name__ == "__main__":
    Motecalo_nums = 10000
    # kernel_args = [X, Y, Z, E, W, r, delta_t, mu, sigma, l]
    X = 0.1
    Y = 0.1
    Z = 0.1
    E = 0
    W = 0.2
    r = 0
    delta_t = 0.1
    mu = 0
    sigma = 0.2
    l = 0.1
    a2 = 0.02
    a3 = 0.1
    kernel_args = [X, Y, Z, E, r, delta_t, mu, sigma, l, a2, a3]



    P_tau = [0.9, 0.95]

    all_W = np.linspace(0, 1, 50000, dtype=np.float64)

    a = pthread_W(kernel_args, all_W, Motecalo_nums, P_tau)
    print(a)