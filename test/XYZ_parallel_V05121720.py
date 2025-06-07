from numba import cuda, prange
import numba
import numpy as np
from Prob import D_Prob
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
from datetime import datetime


a1 = 0.1
a2 = 0.02
r = 0
delta_t = 1
mu = 0
sigma = 0.2

motecalo_nums = 10

min_XYZ = 0
max_XYZ = 5000
size_X = 100
size_Y = 50
size_Z = 50
size_E = 2
size_W = 50




def init_global_XYZEW_V():
    # 将0-10的区间均匀分成X_size份
    X = np.linspace(min_XYZ, max_XYZ, size_X, dtype=np.float64)
    Y = np.linspace(min_XYZ, max_XYZ, size_Y, dtype=np.float64)
    Z = np.linspace(min_XYZ, max_XYZ, size_Z, dtype=np.float64)
    E = np.arange(size_E, dtype=np.int16) 
    W = np.linspace(0, 10, size_W, dtype=np.float64)

    host_array = np.zeros((size_X, size_Y, size_Z, size_E), dtype=np.float64)
    # 创建网格
    X_mesh, Y_mesh, Z_mesh = np.meshgrid(X, Y, Z, indexing='ij')
    
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
    global d_X, d_Y, d_Z, d_E, d_W, d_V, d_V_tp1
    d_V = cuda.to_device(host_array)
    d_V_tp1 = cuda.to_device(host_array.copy())
    d_X = cuda.to_device(X)
    d_Y = cuda.to_device(Y)
    d_Z = cuda.to_device(Z)
    d_E = cuda.to_device(E)
    d_W = cuda.to_device(W)



scale_to_int = float(size_X)/(max_XYZ-min_XYZ)
@cuda.jit(device=True)
def __lookup_V__(d_V, X, Y, Z, E):
    X_int = int(math.floor((X - min_XYZ) * scale_to_int))

    # assert X_int < size_X, "X_int 超出范围" 

    Y_int = int(math.floor((Y - min_XYZ) * scale_to_int))
    Z_int = int(math.floor((Z - min_XYZ) * scale_to_int))
    E_int = int(E)
    return d_V[X_int, Y_int, Z_int, E_int]

# ------------------------------    --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------

def gen_Ptrans_tau(x0, T0, x_end, p, k):
    ax_path = "../data/lc_ax_male.csv"
    bx_path = "../data/lc_bx_male.csv"
    kt_path = "../data/lc_kt_male.csv"
    path = [ax_path, bx_path, kt_path]


    death_Prob = D_Prob()
    death_Prob.read_abk(path)

    interval_P_cache = []

    deal_span = (x_end - x0) * p
    print(type(x_end))
    for i in range(k, deal_span):
            if i == 0:
                interval_P_cache.append(death_Prob.unit_live(x0, T0, i+1, p))
            else:
                interval_P_cache.append(death_Prob.interval_live_P(x0, T0, i, i+1, p))
    
    trans_tau_np = np.array(interval_P_cache)

    T = x_end - x0 - k
    P_tau_eqt = [1 - trans_tau_np[0]]
    P_tau_gep_t = [trans_tau_np[0]]
    for t in range(1, T):
        P_tau_eqt.append((1 - trans_tau_np[t]) * P_tau_gep_t[t-1])
        P_tau_gep_t.append(trans_tau_np[t] * P_tau_gep_t[t-1])

    
    return P_tau_eqt, P_tau_gep_t

# ------------------------------    --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------
@cuda.jit
def XYZEW_kernel(offset, d_results, rng_states, d_P_tau, l, t, a3, d_V, d_X, d_Y, d_Z, d_E, d_W):
    idx = cuda.grid(1) + offset
    if idx >= size_X * size_Y * size_Z * size_E * size_W:
        return
    
    
    index_x = idx // (size_Y * size_Z * size_E * size_W)
    index_y = (idx - index_x * (size_Y * size_Z * size_E * size_W)) // (size_Z * size_E * size_W)
    index_z = (idx - index_x * (size_Y * size_Z * size_E * size_W) - index_y * (size_Z * size_E * size_W)) // (size_E * size_W)
    index_e = (idx - index_x * (size_Y * size_Z * size_E * size_W) - index_y * (size_Z * size_E * size_W) - index_z * (size_E * size_W)) // size_W
    index_w = idx - index_x * (size_Y * size_Z * size_E * size_W) - index_y * (size_Z * size_E * size_W) - index_z * (size_E * size_W) - index_e * size_W
    # print("index_x = %d, index_y = %d, index_z = %d, index_e = %d, index_w = %d" % (index_x, index_y, index_z, index_e, index_w))

    X = d_X[index_x]
    Y = d_Y[index_y]
    Z = d_Z[index_z]
    E = d_E[index_e]
    W = d_W[index_w]



    d_temp = 0.0
    XmW = max((X - W),0)
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
            Y_tp1 = max(XmW , Y - W)
            Z_tp1 = max(a3*XmW, Z)
        elif W > min_ZYt:
            Y_tp1 = max(XmW, 
                        min(Y - W, Y / X * XmW)
                        )
            Z_tp1 = max(a3*XmW, 
                            Z / X * XmW
                        )
            
    P_tau_tp1 = d_P_tau[0] # 这个是P(tau=t+1)时刻的值
    P_tau_gep_tp1 = d_P_tau[1] # 这个是P(tau>=t+1)时刻的值

    E_tp1 = E + W
   
    for i in range(motecalo_nums):
        randomn = xoroshiro128p_normal_float32(rng_states, idx)
        
        X_tp1 = XmW * math.exp( (mu - l - sigma ** 2 / 2) * delta_t + sigma * math.sqrt(delta_t) * randomn)

        if X_tp1 > max_XYZ:
            X_tp1 = max_XYZ
               
        V_tp1 = __lookup_V__(d_V, X_tp1, Y_tp1, Z_tp1, E_tp1)

        d_temp += math.exp(-r * delta_t) * (P_tau_tp1 * max(X_tp1, Y_tp1) + P_tau_gep_tp1 * V_tp1)

    fWt = 0.0
    if W <= min_ZYt:
        fWt = W
    elif W > min_ZYt:
        fWt = W - a1 * (W - min_ZYt)

    if t == 0:
        fWt = 0.0

    d_results[index_x, index_y, index_z, index_e, index_w] = d_temp / motecalo_nums + fWt


# ------------------------------    --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------

@cuda.jit
def V_tp1_kernel(offset, t, d_V_tp1, d_results, d_X, d_Y, d_Z, d_E, d_W):
    idx = cuda.grid(1) + offset
    if idx >= size_X * size_Y * size_Z * size_E:
        return
    
    index_x = idx // (size_Y * size_Z * size_E)
    index_y = (idx - index_x * (size_Y * size_Z * size_E)) // (size_Z * size_E)
    index_z = (idx - index_x * (size_Y * size_Z * size_E) - index_y * (size_Z * size_E)) // size_E
    index_e = idx - index_x * (size_Y * size_Z * size_E) - index_y * (size_Z * size_E) - index_z * size_E

    X = d_X[index_x]
    Y = d_Y[index_y]
    Z = d_Z[index_z]
    E = d_E[index_e]

    max_w = d_results[index_x, index_y, index_z, index_e, 0]
    if t == 0:
        d_V_tp1[index_x, index_y, index_z, index_e] = max_w
        return
    

    for i in range(size_W):
        if Y >= d_W[i] and d_results[index_x, index_y, index_z, index_e, i] > max_w :
            max_w = d_results[index_x, index_y, index_z, index_e, i]
    
    
    d_V_tp1[index_x, index_y, index_z, index_e] = max(
        max(
            Y - a1 * max((Y - min(Z, Y)), 0),
            X
        ), 
        max_w)

# ------------------------------    --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------
def V_iteration(l, t, a3, P_tau_t, d_V, rng_states):
    # start_time = datetime.now()
    # print("iteration %d start time = year month day hour minute second = %d-%d-%d %d:%d:%d" % (t, start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))  
    # d_results = cuda.device_array((size_X, size_Y, size_Z, size_E, size_W), dtype=np.float64)
    # 推荐的方法
    d_results = cuda.to_device(np.zeros((size_X, size_Y, size_Z, size_E, size_W), dtype=np.float64))

    d_P_tau = cuda.to_device(np.array(P_tau_t, dtype=np.float64))
    
    all_treads = size_X * size_Y * size_Z * size_E * size_W
    threads_per_block = 1024
    all_blocks = (all_treads + threads_per_block - 1) // threads_per_block
    

    # rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=1)

    max_blocks_per_grid = 2 ** 16
    for i in range(0, all_blocks, max_blocks_per_grid):
        blocks_per_grid = min(max_blocks_per_grid, all_blocks - i)
        XYZEW_kernel[blocks_per_grid, threads_per_block](i, d_results, rng_states, d_P_tau, l, t, a3,  d_V, d_X, d_Y, d_Z, d_E, d_W)

        cuda.synchronize()
    
    
    # start_time = datetime.now()
    # print("iteration %d result complete = year month day hour minute second = %d-%d-%d %d:%d:%d" % (t, start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))  
    # results = d_results.copy_to_host()
    # print(results)
    ## 第一种方法，在cpu上利用numpy计算最大
    # results = d_results.copy_to_host()
    # V_tp1 = np.max(results, axis=4)

    # 第二种方法，还是在cuda上计算最大值
    all_threads = size_X * size_Y * size_Z * size_E
    threads_per_block = 1024
    all_blocks = (all_threads + threads_per_block - 1) // threads_per_block
    for i in range(0, all_blocks, max_blocks_per_grid):
        blocks_per_grid = min(max_blocks_per_grid, all_blocks - i)
        V_tp1_kernel[blocks_per_grid, threads_per_block](i, t, d_V, d_results, d_X, d_Y, d_Z, d_E, d_W)

        cuda.synchronize()

    
    end_time = datetime.now()
    # print("iteration %d end time = year month day hour minute second = %d-%d-%d %d:%d:%d" % (t, end_time.year, end_time.month, end_time.day, end_time.hour, end_time.minute, end_time.second))
    # print("\n")


    del d_results
    del d_P_tau
    

    # V_tp1 = d_V_tp1.copy_to_host()


# ------------------------------    --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------
def compute_l(l, index, P_tau_eqt, P_tau_gep_t, rng_states):
    # 方式 1：直接在 GPU 内部分配新数组，然后复制数据到新数组
    d_V_tp1.copy_to_device(d_V)
    
    T = len(P_tau_eqt)
    a3 = 1.00/T

    for t in range(T-1, -1, -1):
        P_tau_t = [P_tau_eqt[t], P_tau_gep_t[t]]
        V_iteration(l, t, a3, P_tau_t, d_V_tp1, rng_states)
    
    return d_V_tp1[index[0], index[1], index[2], index[3]] 


# ------------------------------ search l   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------
def preliminary_search_per_i(initial_l, index, step_size, P_tau_eqt, P_tau_gep_t, rng_states, max_iter, initial_investment):
    l = initial_l  # 初始化费用率
    lower_l, upper_l = 0, 1  # 初始化上下界
    for _ in range(max_iter):
        avg_res = compute_l(l, index, P_tau_eqt, P_tau_gep_t, rng_states)
        if avg_res is None:  # 检查 MC 返回值
            print("MC 返回值无效，跳过此迭代")
            break
        print(f"For l = {l:.8f}; 初始值函数 = {avg_res:.8f}; Difference = {abs(initial_investment - avg_res):.8f}")
        
        if avg_res > initial_investment:
            upper_l = l + step_size
            break
        l = max(l - step_size, 0)


    step_size /= 2  # 减少步长以提高精度
    for _ in range(max_iter):
        l += step_size
        
        avg_res = compute_l(l, index, P_tau_eqt, P_tau_gep_t, rng_states)
        if avg_res is None:
            print("MC 返回值无效，跳过此迭代")
            break
        print(f"For l = {l:.8f}; 初始值函数 = {avg_res:.8f}; Difference = {abs(initial_investment - avg_res):.8f}")
        
        if avg_res < initial_investment:
            lower_l = l - step_size
            upper_l = l
            break

    print(f"粗略搜索完成，范围: lower_l = {lower_l:.8f}, upper_l = {upper_l:.8f}")
    return lower_l, upper_l

def fine_search_per_i(lower_l, upper_l, index, ine_step_size, P_tau_eqt, P_tau_gep_t, rng_states, initial_investment):
    best_l = lower_l  
    avg_res = compute_l(best_l, index, P_tau_eqt, P_tau_gep_t, rng_states)

    if avg_res is None:
        print("[fine_search_per_i] MC() 计算返回 None，设置默认值为 0.0")
        avg_res = 0.0
    
    min_difference = abs(initial_investment - avg_res)
    found_small_diff = False  

    # 从 lower_l 开始逐步迭代到 upper_l
    l = lower_l
    while l <= upper_l:
        avg_res = compute_l(l, index, P_tau_eqt, P_tau_gep_t, rng_states)
        if avg_res is None:
            print("[fine_search_per_i] MC() 计算返回 None，设置默认值为 0.0")
            avg_res = 0.0  # 避免 NoneType 错误

        difference = abs(initial_investment - avg_res)
        print(f"[fine_search_per_i] For l = {l:.8f}; 初始值函数 = {avg_res:.8f}; Difference = {difference:.8f}")

        if difference < 0.00005:
            best_l = l
            min_difference = difference
            print(f"[fine_search_per_i] 提前找到最佳费用率 l = {best_l:.8f}; 最小差异 = {min_difference:.8f}")
            break

        if difference < 0.0003:
            found_small_diff = True

        if found_small_diff and difference > 0.002:
            print("[fine_search_per_i] 差异超过 0.002 后停止搜索")
            break

        if difference < min_difference:
            min_difference = difference
            best_l = l

        l += fine_step_size  

    print(f"[fine_search_per_i] 精细搜索完成: 最佳费用率 l = {best_l:.8f}, 最小差异 = {min_difference:.8f}")
    return best_l, min_difference






if __name__ == "__main__":
    x0 = 67
    x_end = 92
    p, q = 1, 1
    initial_investment = 100.0

    delta_l = 0
    initial_l = 0.01
    step_size = 0.001
    max_iter=10

    init_global_XYZEW_V()

    all_treads = size_X * size_Y * size_Z * size_E * size_W
    threads_per_block = 1024
    all_blocks = (all_treads + threads_per_block - 1) // threads_per_block
    rng_states = create_xoroshiro128p_states(threads_per_block * all_blocks, seed=1)


    for k in range(0,25):
        print("\n k = %d" % k)
        P_tau_eqt, P_tau_gep_t = gen_Ptrans_tau(x0, 2020, x_end, p, k)


        X_index = int(size_X/2)
        Y_index = int(size_Y/2)
        a3 = 1.00/(x_end - x0- k)
        Z_index = int(math.floor((a3*initial_investment - min_XYZ) * scale_to_int))
        index = [X_index, Y_index, Z_index, 0]


        # 调用粗略搜索函数
        lower_l, upper_l = preliminary_search_per_i(initial_l, index, step_size, P_tau_eqt, P_tau_gep_t, rng_states, max_iter, initial_investment)
        print(f"粗略搜索结果: lower_l = {lower_l:.8f}, upper_l = {upper_l:.8f}")

        if lower_l is not None and upper_l is not None:
            fine_step_size = 0.0000001
            best_l, min_difference = fine_search_per_i(
                lower_l, upper_l, index, fine_step_size, P_tau_eqt, P_tau_gep_t, rng_states, initial_investment
            )
            delta_l = best_l - initial_l
            initial_l = best_l
            result_str = f"最佳费用率 l = {best_l:.8f}，最小差异 = {min_difference:.8f}\n"
            print(result_str)