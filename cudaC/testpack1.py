import compute_l
import time
compute_l.init_global_config(
        0, 200, 201,
        0, 200, 201,
        0, 100, 51,
        0, 1,   2,
        0, 200, 201,
        0.15, 0.025, 0.05, 0.05, 0.2, 100000, 1, 100.0)

compute_l.init_global_XYZEW_V()
# CUDA_VISIBLE_DEVICES=0 nohup python testpack1.py >> male1_test.log 2>&1 &

initial_l = 0.109562
# initial_l = 0.109544

live_list = [0.84799892]
# live_list = [0.87653274]
print(f'T = {len(live_list)}, initial_l is {initial_l}')

time_start = time.time()
out = compute_l.compute_l(initial_l, live_list)

# compute_l.reset_Vtp1()

# compute_l.compute_l(initial_l, live_list)

# compute_l.reset_Vtp1()

# compute_l.compute_l(initial_l, live_list)

time_end = time.time()
print("time used: ", time_end - time_start)
print("out is ", out)

compute_l.clean_global_XYZEW_V()

