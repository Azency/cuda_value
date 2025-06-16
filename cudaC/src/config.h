// 定义常量宏 start
#define A1 0.15f
#define A2 0.025f
#define R 0.05f

#define MU 0.05f
#define SIGMA 0.2f
#define MOTECALO_NUMS 10000

#define MIN_XYZ 0
#define MAX_X 100
#define MAX_Y 100
#define MAX_Z 100
#define MAX_W 100


#define SIZE_X 41
#define SIZE_Y 41
#define SIZE_Z 41
#define SIZE_E 2
#define SIZE_W 41
// 定义常量宏 end


#define SCALE_TO_INT_X ((float)(SIZE_X-1) / (MAX_X - MIN_XYZ))
#define SCALE_TO_INT_Y ((float)(SIZE_Y-1) / (MAX_Y - MIN_XYZ))
#define SCALE_TO_INT_Z ((float)(SIZE_Z-1) / (MAX_Z - MIN_XYZ))

#define sXYZEW (SIZE_X*SIZE_Y*SIZE_Z*SIZE_E*SIZE_W)
#define sYZEW (SIZE_Y*SIZE_Z*SIZE_E*SIZE_W)
#define sZEW (SIZE_Z*SIZE_E*SIZE_W)
#define sEW (SIZE_E*SIZE_W)
#define sW (SIZE_W)

#define sXYZE (SIZE_X*SIZE_Y*SIZE_Z*SIZE_E)
#define sYZE (SIZE_Y*SIZE_Z*SIZE_E)
#define sZE (SIZE_Z*SIZE_E)
#define sE (SIZE_E)


#define X0 67
#define X_END 92
#define P 1
#define Q 1
#define INITIAL_INVESTMENT 100.0f
#define DELTA_T 1.0f/P

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

extern float *d_X, *d_Y, *d_Z, *d_W, *d_V, *d_V_tp1, *d_results;
extern int *d_E;

__constant__ float *d_d_X, *d_d_Y, *d_d_Z, *d_d_W, *d_d_V, *d_d_V_tp1, *d_d_results;
__constant__ int *d_d_E;