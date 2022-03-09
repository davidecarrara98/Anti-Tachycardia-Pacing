import sys

import numpy as np

from BayesianOptimizer2D import BayesOptimizer2D

sys.path.append("scripts")
from utils2 import load_nu, custom_loss, l2_int

patient = 3
nu2 = load_nu(error_type = "MSE")[patient]
opt = BayesOptimizer2D(nu=nu2, refined_grid=True, niter=40, load_all=True, min_iter=0,
                       error_function=custom_loss)
t, dur, mod = opt.optimize()

inds = np.where(np.abs(opt.final_pred) < np.quantile(opt.final_pred, 0.05))
vals = opt.prediction_grid[inds]
fin_ind = np.where(vals[:,1] == np.min(vals[:,1]))
chosen = vals[fin_ind]