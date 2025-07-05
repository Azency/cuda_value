#include "cuda_value.h"


// 导出到python的函数
extern "C" 
float pycompute_l(float l, float * trans_tau_d, int T) {
    return compute_l(l, trans_tau_d, T);
}


extern "C"
void pyinit_global_XYZEW_V() {
    init_global_XYZEW_V();
}

extern "C"
void pyclean_global_XYZEW_V() {
    clean_global_XYZEW_V();
}

extern "C"
void pyreset_Vtp1() {
    reset_Vtp1();
}

extern "C"
void pyinit_global_config(
    float min_X, float max_X, int size_X,
    float min_Y, float max_Y, int size_Y,
    float min_Z, float max_Z, int size_Z,
    int min_E, int max_E, int size_E,
    float min_W, float max_W, int size_W,
    float a1, float a2, float r, float mu, float sigma, int motecalo_nums, float p, float initial_investment
) {
    init_global_config(
        min_X, max_X, size_X, 
        min_Y, max_Y, size_Y, 
        min_Z, max_Z, size_Z, 
        min_E, max_E, size_E, 
        min_W, max_W, size_W, 
        a1, a2, r, mu, sigma, motecalo_nums, p, initial_investment);
}