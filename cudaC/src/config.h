extern int h_MIN_X, h_MIN_Y, h_MIN_Z, h_MIN_W;
extern int h_MAX_X, h_MAX_Y, h_MAX_Z, h_MAX_W;
extern int h_SIZE_X, h_SIZE_Y, h_SIZE_Z, h_SIZE_E, h_SIZE_W;
extern int h_sXYZEW, h_sYZEW, h_sZEW, h_sEW, h_sW;
extern int h_sXYZE, h_sYZE, h_sZE, h_sE;
extern float h_SCALE_TO_INT_X, h_SCALE_TO_INT_Y, h_SCALE_TO_INT_Z;

__constant__ int d_MIN_X, d_MIN_Y, d_MIN_Z, d_MIN_W;
__constant__ int d_MAX_X, d_MAX_Y, d_MAX_Z, d_MAX_W;
__constant__ int d_SIZE_X, d_SIZE_Y, d_SIZE_Z, d_SIZE_E, d_SIZE_W;
__constant__ int d_sXYZEW, d_sYZEW, d_sZEW, d_sEW, d_sW;
__constant__ int d_sXYZE, d_sYZE, d_sZE, d_sE;
__constant__ float d_SCALE_TO_INT_X, d_SCALE_TO_INT_Y, d_SCALE_TO_INT_Z;


extern float h_A1,h_P, h_INITIAL_INVESTMENT, h_DELTA_T;
__constant__ float d_A1, d_A2, d_R, d_MU, d_SIGMA, d_P, d_INITIAL_INVESTMENT, d_DELTA_T;
__constant__ int d_MOTECALO_NUMS;

extern float *d_X, *d_Y, *d_Z, *d_W, *d_V, *d_V_tp1, *d_results;
extern int *d_E;

__constant__ float *d_d_X, *d_d_Y, *d_d_Z, *d_d_W, *d_d_V, *d_d_V_tp1, *d_d_results;
__constant__ int *d_d_E;