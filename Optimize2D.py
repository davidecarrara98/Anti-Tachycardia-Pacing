import sys
import numpy as np
from BayesianOptimizer2D import BayesOptimizer2D

sys.path.append("scripts")
from utils2 import load_nu, custom_loss, l2_int, l2_int_restricted

patient = 1
nu2 = load_nu(error_type = "MSE")[patient]
opt = BayesOptimizer2D(nu=nu2, refined_grid=True, niter=5, load_all=True, min_iter=10,
                       error_function=l2_int_restricted)
t, dur, mod = opt.optimize()