import compute_l

compute_l.init_global_config(
    0, 100, 41, 
    0, 100, 41, 
    0, 100, 41, 
    0, 2, 2, 
    0, 100, 41, 
    0.15, 0.025, 0.05, 0.05, 0.2, 10000, 1, 100)

compute_l.init_global_XYZEW_V()


out = compute_l.compute_l(0.00, [0.95227777, 0.9458399, 0.938519, 0.93016787, 0.92060485, 0.9096251, 0.89702214, 0.88261673, 0.86628806, 0.84799892])

print("out is ", out)

compute_l.reset_Vtp1()

out2 = compute_l.compute_l(0.00, [0.95227777, 0.9458399, 0.938519, 0.93016787, 0.92060485, 0.9096251, 0.89702214, 0.88261673, 0.86628806, 0.84799892])

print("out2 is ", out2)


compute_l.clean_global_XYZEW_V()

