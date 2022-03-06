import numpy as np
import matplotlib.pyplot as plt
from BayesianOptimizer2D import BayesOptimizer2D
import sys

sys.path.append("scripts")
from utils2 import load_nu

patient = 3
nu2 = load_nu(error_type = "MSE")[patient]
opt = BayesOptimizer2D(nu=nu2, refined_grid=True, niter=0, load_all=True, min_iter=0)
t, dur, mod = opt.optimize()

#t_min, t_max, dur_min, dur_max = 450, 525, 0.005, 10
#t_vec = np.arange(t_min, t_max, dur_min * 10)
#d_vec = np.arange(dur_min, dur_max, dur_min * 10)
#t_vec, d_vec = np.meshgrid(t_vec, d_vec)

#prediction_grid = np.mgrid[t_min:t_max:dur_min*10, dur_min:dur_max:dur_min*10].reshape(2, -1).T
#yp, vp = mod.predict(prediction_grid / np.array([10, 1]))
#al = yp.reshape(1500, 200).T
#plt.figure(figsize=(20,20), dpi=64)
#ax = plt.axes(projection='3d')
#ax.view_init(elev = 20, azim=-40)
#ax.plot_surface(t_vec, d_vec, al, rstride=1, cstride=1, cmap='seismic',
#                antialiased=False, edgecolor='none')
#ax.set_title('surface', fontsize=60)
#ax.set_xlabel('Activation Time', fontsize=35, labelpad=30)
#ax.set_ylabel('Duration', fontsize=35, labelpad=30)
#ax.set_zlabel('MSE', fontsize=35, labelpad=30)
# Set tick font size
#for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
#	label.set_fontsize(30)
#plt.show()

