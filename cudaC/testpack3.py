import compute_l
import time
compute_l.init_global_config(
    0, 300, 181,
        0, 300, 181,
        0, 100, 31,
        0, 1,   2,
        0, 100, 31,
        0.15, 0.025, 0.05, 0.05, 0.2, 100000, 1, 100.0)

compute_l.init_global_XYZEW_V()

# CUDA_VISIBLE_DEVICES=3 nohup python testpack3.py >> male3_test.log 2>&1 &

# initial_l = 0.072361
initial_l = 0.071304

# live_list = [0.88261673, 0.86628806, 0.84799892]
live_list = [0.90653058, 0.89236242, 0.87653274]

print(f'T = {len(live_list)}, initial_l is {initial_l}')

time_start = time.time()
out = compute_l.compute_l(initial_l, live_list)


time_end = time.time()
print("time used: ", time_end - time_start)
print("out is ", out)

compute_l.clean_global_XYZEW_V()

