import compute_l
import time
compute_l.init_global_config(
        # 0, 1500, 451,
        # 0, 1500, 451,
        # 0, 100, 16,
        # 0, 1,   2,
        # 0, 500, 151,
        # 0.15, 0.025, 0.05, 0.05, 0.2, 100000, 1, 100.0
        # 
        0, 1500, 451,
        0, 1500, 451,
        0, 100, 31,
        0, 1,   2,
        0, 500, 151,
        0.15, 0.025, 0.05, 0.05, 0.2, 100000, 1, 100.0)

compute_l.init_global_XYZEW_V()
# CUDA_VISIBLE_DEVICES=1 nohup python testpack15.py >> male15_test.log 2>&1 &

initial_l = 0.034991
# initial_l = 0.033966

live_list = [0.97422256, 0.97100599, 0.96725252, 0.96291495, 0.95794281, 0.95227777, 0.9458399,  0.938519, 0.93016787, 0.92060485, 0.9096251, 0.89702214, 0.88261673, 0.86628806, 0.84799892]
# live_list = [0.98255486, 0.98028799, 0.97760967, 0.9744482,  0.97071873, 0.96632442, 0.96115395, 0.95508318, 0.9479776,  0.93969821, 0.93010926, 0.91908749, 0.90653058, 0.89236242, 0.87653274]
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

