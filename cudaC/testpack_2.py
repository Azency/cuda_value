import compute_l
import time
compute_l.init_global_config(
    0, 1000, 1001, 
    0, 1000, 101, 
    0, 100, 11, 
    0, 1, 2, 
    0, 1000, 101, 
    0.15, 0.025, 0.05, 0.05, 0.2, 1000, 1, 100)

compute_l.init_global_XYZEW_V()

time_start = time.time()
out = compute_l.compute_l(0.039, [0.95227777, 0.9458399, 0.938519, 0.93016787, 0.92060485, 0.9096251, 0.89702214, 0.88261673, 0.86628806, 0.84799892])
time_end = time.time()
print("time used: ", time_end - time_start)
print("out is ", out)

compute_l.clean_global_XYZEW_V()

