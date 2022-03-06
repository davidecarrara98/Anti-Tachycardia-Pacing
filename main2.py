import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("scripts")
from utils2 import save_first_450, generate_last_350, load_nu

patient = 3
nu2 = load_nu(error_type = "MSE")[patient]

grid_name = "refine"
grid = True if grid_name == "refine" else False

#save_first_450(nu2, refined_grid=grid)
#500.470 - 7.275
new_signal = generate_last_350(nu2, ICD_time=450, ICD_duration=0.005, refined_grid=grid)
ori_signal = np.load("signals_3_patients.npy")

# add save

plt.plot(ori_signal[patient-1,1,:],ori_signal[patient-1,0,:],"g")
plt.plot(new_signal[0,1,:],new_signal[0,0,:],"b")
plt.plot([450,450],[-1.5,1],"r")
plt.show()