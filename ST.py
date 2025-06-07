import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(N, T):
    """
    模拟布朗运动
    N : 分区数量
    T : 总时间
    h : 时间步长
    """
    dt = T/N
    t = np.linspace(0, T, N)
    W = np.zeros(N)
    W[1:] = np.cumsum(np.sqrt(dt)*np.random.standard_normal(size=N-1))  # 累积和
    return t, W


def St_motion(N, T, S0, r, l, sigma):
    """
    模拟股票的随机过程
    N : 分区数量
    T : 总时间
    h : 时间步长
    """
    dt = T/N
    t = np.linspace(0, T, N)
    W = np.zeros(N)
    lgSt = np.zeros(N)
    W[1:] = np.cumsum(np.sqrt(dt)*np.random.standard_normal(size=N-1))  # 累积和
    lgSt[1:] = np.cumsum((r - l - 0.5 * sigma**2) * dt + sigma * (W[1:] - W[0:-1]))  # 迭代生成
    St = S0*np.exp(lgSt)
    return t, St

def intg_item(N, p, T, S0, r, l, g, sigma) -> list:
    """
    模拟股票的随机过程
    N : 分区数量
    T : 总时间
    h : 时间步长

    S0 : 初始值
    r : 
    l :
    g : 
    sigma :
    """
    dt = T/(N*p)
    t = np.linspace(0, T/p, N)
    W = np.zeros(N)
    lgSt = np.zeros(N)
    W[1:] = np.cumsum(np.sqrt(dt)*np.random.standard_normal(size=N-1))  # 累积和
    lgSt[0] = np.log(S0)
    lgSt[1:] = np.cumsum((r - l - 0.5 * sigma**2) * dt + sigma * (W[1:] - W[0:-1]))  # 迭代生成
    St = S0 * np.exp(lgSt)
    # print("st",St)

    final_int = np.zeros(N)
    for i in range(N):
        final_int[i] = np.exp(-r * t[i]) * np.maximum(np.exp(g*t[i]) - St[i], 0)
        
        # final_int[i] = np.exp(-r * t[i]) * np.maximum(np.exp(g*t[i]), St[i])
        # print("exp(-r * t[i])",np.exp(-r * t[i]))
        # print("maximum",np.maximum(np.exp(g*t[i]) - St[i], 0))
        # print("final_int[i]",final_int[i])

    return t, final_int, St