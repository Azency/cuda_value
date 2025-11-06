import compute_l
import time
compute_l.init_global_config(
        0, 500, 251,
        0, 500, 251,
        0, 100, 26,
        0, 1,   2,
        0, 500, 251,
        0.15, 0.025, 0.05, 0.05, 0.2, 100000, 1, 100.0)

compute_l.init_global_XYZEW_V()
# CUDA_VISIBLE_DEVICES=1 nohup python testpack5.py >> male5_test.log 2>&1 &

initial_l = 0.057128
# initial_l = 0.055466

live_list = [0.9096251, 0.89702214, 0.88261673, 0.86628806, 0.84799892]
# live_list = [0.93010926, 0.91908749, 0.90653058, 0.89236242, 0.87653274]
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

