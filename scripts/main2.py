import numpy as np
import utils2

nu2 = 0.0124
grid = 'fine'
utils2.save_first_450(nu2=0.0124, refined_grid=True)

signal = np.load(f'../First_450/Signal_patient_{nu2 : .6f}_{grid}.npy')
Ut = np.load(f'../First_450/Ut_patient_{nu2 : .6f}_{grid}.npy')
Wt = np.load(f'../First_450/Wt_patient_{nu2 : .6f}_{grid}.npy')
Nu2 = np.load(f'../First_450/Nu2_patient_{nu2 : .6f}_{grid}.npy')

