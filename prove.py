import sys

import numpy as np

from BayesianOptimizer2D import BayesOptimizer2D

sys.path.append("scripts")
from utils2 import load_nu, custom_loss, l2_int

patient = 1
nu2 = load_nu(error_type = "MSE")[patient]
opt = BayesOptimizer2D(nu=nu2, refined_grid=True, niter=40, load_all=True, min_iter=5,
                       error_function=l2_int)
t, dur, mod = opt.optimize()

inds = np.where(np.abs(opt.final_pred) < np.quantile(np.abs(opt.final_pred), 0.03))
vals = opt.prediction_grid[inds]
fin_ind = np.where(vals[:,1] == np.min(vals[:,1]))
chosen = vals[fin_ind]