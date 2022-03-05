import numpy as np
import matplotlib.pyplot as plt
from BayesianOptimizer2D import BayesOptimizer2D
import sys

sys.path.append("scripts")
from utils2 import load_nu

patient = 3
nu2 = load_nu(error_type = "MSE")[patient]
opt = BayesOptimizer2D(nu=nu2, refined_grid=True, niter=20, load_all=True, min_iter=0)
t, dur, mod = opt.optimize()