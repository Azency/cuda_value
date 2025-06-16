from numba import cuda, prange
import numba
import numpy as np
from Prob import D_Prob
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
from datetime import datetim



def gen_Ptrans_tau(x0, T0, x_end, p, k):
    ax_path = "./data/lc_ax_male.csv"
    bx_path = "./data/lc_bx_male.csv"
    kt_path = "./data/lc_kt_male.csv"
    path = [ax_path, bx_path, kt_path]


    death_Prob = D_Prob()
    death_Prob.read_abk(path)

    interval_P_cache = []

    deal_span = (x_end - x0) * p
    for i in range(k, deal_span):
            if i == 0:
                interval_P_cache.append(death_Prob.unit_live(x0, T0, i+1, p))
            else:
                interval_P_cache.append(death_Prob.interval_live_P(x0, T0, i, i+1, p))
    
    trans_tau_np = np.array(interval_P_cache)#存活概率
    assert trans_tau_np.__len__() == deal_span - k
    trans_tau_d = cuda.to_device(trans_tau_np)
    
    return trans_tau_d

a1 = 0.15
a2 = 0.025
r = 0.05
mu = 0.05
sigma = 0.2
motecalo_nums = 10


x0 = 67
x_end =92
p, q = 1, 1
delta_t = 1/p
initial_investment = 100.0


min_XYZ = 0
max_X = 100
max_Y = 100
max_Z = 100
max_W = 100

size_X = 21
size_Y = 21
size_Z = 21
size_E = 2
size_W = 21

sXYZEW = size_X * size_Y * size_Z * size_E * size_W
sYZEW = size_Y * size_Z * size_E * size_W
sZEW = size_Z * size_E * size_W
sEW = size_E * size_W
sW = size_W

sXYZE = size_X * size_Y * size_Z * size_E
sYZE = size_Y * size_Z * size_E
sZE = size_Z * size_E
sE = size_E

scale_to_int_X = float(size_X-1)/(max_X-min_XYZ)
scale_to_int_Y = float(size_Y-1)/(max_Y-min_XYZ)
scale_to_int_Z = float(size_Z-1)/(max_Z-min_XYZ)



def init_global_XYZEW_V():
    # 将0-10的区间均匀分成X_size份
    X = np.linspace(min_XYZ, max_X, size_X, dtype=np.float64)
    Y = np.linspace(min_XYZ, max_Y, size_Y, dtype=np.float64)
    Z = np.linspace(min_XYZ, max_Z, size_Z, dtype=np.float64)
    E = np.arange(size_E, dtype=np.int16) 
    W = np.linspace(min_XYZ, max_W, size_W, dtype=np.float64)

    host_array = np.zeros((size_X, size_Y, size_Z, size_E), dtype=np.float64)
    # 创建网格
    X_mesh, Y_mesh, Z_mesh = np.meshgrid(X, Y, Z, indexing='ij')
    
    # 计算min(Z, Y)
    min_ZY = np.minimum(Z_mesh, Y_mesh)
    
    # 计算Y - a1*(Y - min(Z, Y))
    term = np.where(Y_mesh <= min_ZY, Y_mesh, Y_mesh - a1 * (Y_mesh - min_ZY))
    
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



# @cuda.jit(device=True)
# def __lookup_V__(d_V, X, Y, Z, E):
#     X_int = int(math.floor((X - min_XYZ) * scale_to_int))
#     Y_int = int(math.floor((Y - min_XYZ) * scale_to_int))

@cuda.jit(device=True)
def __lookup_V__(
    d_V, X, Y, Z, E):

    # 计算X方向参数
    X_int = int(math.floor((X - min_XYZ) * scale_to_int_X))
    x_base = min_XYZ + X_int / scale_to_int_X
    dx = (X - x_base) / scale_to_int_X if scale_to_int_X != 0 else 0.0

    # 计算Y方向参数
    Y_int = int(math.floor((Y - min_XYZ) * scale_to_int_Y))
    y_base = min_XYZ + Y_int / scale_to_int_Y
    dy = (Y - y_base) / scale_to_int_Y if scale_to_int_Y != 0 else 0.0

    # 计算Z方向参数
    # Z_int = int(math.floor((Z - min_XYZ) * scale_to_int_Z))
    # z_base = min_XYZ + Z_int / scale_to_int_Z
    # dz = (Z - z_base) / scale_to_int_Z if scale_to_int_Z != 0 else 0.0
    Z_int = int(math.floor((Z - min_XYZ) * scale_to_int_Z))
    dz = 0
    
    E_int = int(E)

    # 处理边界索引
    X_p1 = min(X_int + 1, size_X - 1)
    Y_p1 = min(Y_int + 1, size_Y - 1)
    Z_p1 = min(Z_int + 1, size_Z - 1)

    # 获取8个邻近点的值
    v0000 = d_V[X_int, Y_int, Z_int, E_int]
    v1000 = d_V[X_p1, Y_int, Z_int, E_int]
    v0100 = d_V[X_int, Y_p1, Z_int, E_int]
    v1100 = d_V[X_p1, Y_p1, Z_int, E_int]
    v0010 = d_V[X_int, Y_int, Z_p1, E_int]
    v1010 = d_V[X_p1, Y_int, Z_p1, E_int]
    v0110 = d_V[X_int, Y_p1, Z_p1, E_int]
    v1110 = d_V[X_p1, Y_p1, Z_p1, E_int]
    # 三维双线性插值（三次线性插值组合）
    return (
        (1 - dx) * (1 - dy) * (1 - dz) * v0000 +
        dx * (1 - dy) * (1 - dz) * v1000 +
        (1 - dx) * dy * (1 - dz) * v0100 +
        dx * dy * (1 - dz) * v1100 +
        (1 - dx) * (1 - dy) * dz * v0010 +
        dx * (1 - dy) * dz * v1010 +
        (1 - dx) * dy * dz * v0110 +
        dx * dy * dz * v1110
    )


#     Z_int = int(math.floor((Z - min_XYZ) * scale_to_int))
#     E_int = int(E)
#     return d_V[X_int, Y_int, Z_int, E_int]


@cuda.jit
def XYZEW_kernel(offset, d_results, rng_states, d_P_tau, l, a3, t, d_V, d_X, d_Y, d_Z, d_E, d_W):
    idx = cuda.grid(1) + offset
    if idx >= sXYZEW:
        return
    
    index_x = idx / sYZEW
    remainder = idx % sYZEW
    index_y = remainder / sZEW
    remainder = remainder % sZEW
    index_z = remainder / sEW
    remainder = remainder % sEW
    index_e = remainder / sW
    index_w = remainder % sW

    X = d_X[index_x]
    Y = d_Y[index_y]
    Z = d_Z[index_z]
    E = d_E[index_e]
    W = d_W[index_w]

    min_ZYt = min(Z, Y)

    E_tp1 = 1 * (E + W == 0)



    if W <= Y:
        d_temp = 0.0
        XmW = max((X - W),0)
        min_ZYt = min(Z, Y)
        E_tp1 = 0 if E + W==0 else 1
        if W == 0:
            Y_tp1 = (1 - E_tp1) * max((1 + a2)*Y, XmW) + E_tp1 * max(Y, XmW)
            Z_tp1 = (1 - E_tp1) * max(a3*Y_tp1, a3*XmW) + E_tp1 * max(Z, a3*XmW)

        elif W >0 :
            if W <= min_ZYt:
                Y_tp1 = max(Y - W, XmW)
                Z_tp1 = max(Z, a3*XmW)
            elif W > min_ZYt:
                Y_tp1 = 0 if X == 0 else max(min(Y - W, Y / X * XmW), XmW)
                Z_tp1 = 0 if X == 0 else max(Z / X * XmW, a3*XmW)
        
        Y_tp1 = min(Y_tp1, max_Y)
        Z_tp1 = min(Z_tp1, max_Z)

        P_tau_tp1 = d_P_tau[0] # 这个是P(tau=t+1)时刻的值
        P_tau_gep_tp1 = d_P_tau[1] # 这个是P(tau>=t+1)时刻的值
    
        for i in range(motecalo_nums):
            random = xoroshiro128p_normal_float32(rng_states, idx)
            # random = 0.5
            X_tp1 = XmW * math.exp( (mu - l - 0.5 * sigma ** 2) * delta_t + sigma * math.sqrt(delta_t) * random)
            
            X_tp1 = min(X_tp1, max_X)


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
        if d_W[i] <=Y  and d_results[index_x, index_y, index_z, index_e, i] > max_w :
            max_w = d_results[index_x, index_y, index_z, index_e, i]
    
    
    d_V_tp1[index_x, index_y, index_z, index_e] = max(
        max(Y - a1 * (Y - min(Z, Y)),  X), 
        max_w)

# ------------------------------    --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------
def V_iteration(l, t, a3, P_tau_t, d_V, rng_states):
    start_time = datetime.now()
    print("iteration %d start time = year month day hour minute second = %d-%d-%d %d:%d:%d" % (t, start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))  
    # d_results = cuda.device_array((size_X, size_Y, size_Z, size_E, size_W), dtype=np.float64)
    # 推荐的方法
    d_results = cuda.to_device(np.zeros((size_X, size_Y, size_Z, size_E, size_W), dtype=np.float64))

    d_P_tau = cuda.to_device(np.array(P_tau_t, dtype=np.float64))
    
    all_treads = size_X * size_Y * size_Z * size_E * size_W
    threads_per_block = 1024
    all_blocks = (all_treads + threads_per_block - 1) // threads_per_block
    

    # rng_states = create_xoroshiro128p_states(threads_per_block * all_blocks, seed=t)

    max_blocks_per_grid = 2 ** 16
    for i in range(0, all_blocks, max_blocks_per_grid):
        blocks_per_grid = min(max_blocks_per_grid, all_blocks - i)
        XYZEW_kernel[blocks_per_grid, threads_per_block](i, d_results, rng_states, d_P_tau, l, t, a3,  d_V, d_X, d_Y, d_Z, d_E, d_W)

        cuda.synchronize()
    start_time = datetime.now()
    print("iteration %d result complete = year month day hour minute second = %d-%d-%d %d:%d:%d" % (t, start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))  


    # 第二种方法，还是在cuda上计算最大值
    all_threads = size_X * size_Y * size_Z * size_E
    threads_per_block = 1024
    all_blocks = (all_threads + threads_per_block - 1) // threads_per_block
    for i in range(0, all_blocks, max_blocks_per_grid):
        blocks_per_grid = min(max_blocks_per_grid, all_blocks - i)
        V_tp1_kernel[blocks_per_grid, threads_per_block](i, t, d_V, d_results, d_X, d_Y, d_Z, d_E, d_W)

        cuda.synchronize()

    
    end_time = datetime.now()
    print("iteration %d end time = year month day hour minute second = %d-%d-%d %d:%d:%d" % (t, end_time.year, end_time.month, end_time.day, end_time.hour, end_time.minute, end_time.second))
    print("\n")


    del d_results
    del d_P_tau
    
# ------------------------------    --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------   --------------------------------
def compute_l(l, index, trans_tau_d, rng_states):
    # 方式 1：直接在 GPU 内部分配新数组，然后复制数据到新数组
    d_V_tp1.copy_to_device(d_V)
    
    T = len(trans_tau_d)
    a3 = 1.00/(T/p)
    print(f"T ={T}, l={l}, a3={a3}")

    for t in range(T-1, -1, -1):
        P_tau_t = [1-trans_tau_d[t], trans_tau_d[t]]#trans_tau_d存活概率
        V_iteration(l, t, a3, P_tau_t, d_V_tp1, rng_states)

    print(f"索引1,{index[0][0], index[0][1], index[0][2], index[0][3]}")
    print(f"对应账户值是：{d_X[index[0][0]]}, {d_Y[index[0][1]]}, {d_Z[index[0][2]]}, {d_E[index[0][3]]}")
    print(f"索引2,{index[1][0], index[1][1], index[1][2], index[1][3]}")
    print(f"对应账户值是：{d_X[index[1][0]]}, {d_Y[index[1][1]]}, {d_Z[index[1][2]]}, {d_E[index[1][3]]}")
    print(f"索引3,{index[2]}")
    
    output_1 = d_V_tp1[index[0][0], index[0][1], index[0][2], index[0][3]]
    output_2 = d_V_tp1[index[1][0], index[1][1], index[1][2], index[1][3]]
    output = output_1 + (output_2 - output_1)*index[2]
    print(output_1, output_2, output)
    return output


def preliminary_search_per_i(initial_l, index, step_size, trans_tau_d, rng_states, max_iter, initial_investment):
    l = initial_l  # 初始化费用率
    lower_l, upper_l = 0, 1  # 初始化上下界
    for _ in range(max_iter):
        avg_res = compute_l(l, index, trans_tau_d, rng_states)
        if avg_res > initial_investment:
            upper_l = l + step_size
            break
        l = max(l - step_size, 0)


    step_size /= 2  # 减少步长以提高精度
    for _ in range(max_iter):
        l += step_size
        
        avg_res = compute_l(l, index, trans_tau_d, rng_states)
        if avg_res < initial_investment:
            lower_l = l - step_size
            upper_l = l
            break

    print(f"粗略搜索完成，范围: lower_l = {lower_l:.8f}, upper_l = {upper_l:.8f}")
    return lower_l, upper_l

def fine_search_per_i(lower_l, upper_l, index, fine_step_size, trans_tau_d, rng_states, initial_investment):
    best_l = lower_l  
    avg_res = compute_l(best_l, index, trans_tau_d, rng_states)

    if avg_res is None:
        print("[fine_search_per_i] MC() 计算返回 None, 设置默认值为 0.0")
        avg_res = 0.0
    
    min_difference = abs(initial_investment - avg_res)
    found_small_diff = False  

    # 从 lower_l 开始逐步迭代到 upper_l
    l = lower_l
    while l <= upper_l:
        avg_res = compute_l(l, index, trans_tau_d, rng_states)
        if avg_res is None:
            print("[fine_search_per_i] MC() 计算返回 None, 设置默认值为 0.0")
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
    delta_l = 0
    initial_l = 0.04
    step_size = 0.001
    max_iter= 1

    k=15
    a3 = 1.00/(x_end - x0 - k)
    max_Z = a3*max_X/p
    
    init_global_XYZEW_V()
    
    all_treads = size_X * size_Y * size_Z * size_E * size_W
    threads_per_block = 1024
    all_blocks = (all_treads + threads_per_block - 1) // threads_per_block
    rng_states = create_xoroshiro128p_states(threads_per_block * all_blocks, seed=1)
    scale_to_int_Z = float(size_Z-1)/(max_Z-min_XYZ)
    
    for k in range(15,25):
        print("\n k = %d" % k)
        # a3 = 1.00/(x_end - x0 - k)
        # max_Z = a3*max_X/p

        ####寻找计算初始值的指标索引######################################################
        X_index = int(math.floor((initial_investment - min_XYZ) * scale_to_int_X))
        Y_index = int(math.floor((initial_investment - min_XYZ) * scale_to_int_Y))
        Z_index_1 = int(math.floor((a3*initial_investment - min_XYZ) * scale_to_int_Z))
        delta_z = (a3*initial_investment - min_XYZ) * scale_to_int_Z - Z_index_1
        Z_index_2 = int(min(Z_index_1+1, size_Z-1))
        print(f'Z_index_1 = {Z_index_1}, Z_index_2 = {Z_index_2}, a3*initial_investment = {a3*initial_investment}, scale_to_int_Z = {scale_to_int_Z}, delta_z = {delta_z}')
        index = [[X_index, Y_index, Z_index_1, 0], [X_index, Y_index, Z_index_2, 0], delta_z]


        trans_tau_d = gen_Ptrans_tau(x0, 2020, x_end, p, k)
        # 调用粗略搜索函数
        lower_l, upper_l = preliminary_search_per_i(initial_l, index, step_size, trans_tau_d, rng_states, max_iter, initial_investment)
        print(f"索引{25-k},粗略搜索结果: lower_l = {lower_l:.8f}, upper_l = {upper_l:.8f}")

        if lower_l is not None and upper_l is not None:
            fine_step_size = 0.0000001
            best_l, min_difference = fine_search_per_i(
                lower_l, upper_l, index, fine_step_size, trans_tau_d, rng_states, initial_investment
            )
            delta_l = best_l - initial_l
            initial_l = best_l
            result_str = f"索引{25-k}, 最佳费用率 l = {best_l:.8f}，最小差异 = {min_difference:.8f}\n"
            print(result_str)


