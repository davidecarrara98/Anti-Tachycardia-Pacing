import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("scripts")
from utils2 import save_first_450, generate_last_350, load_nu, l2_int

patient = 3
nu2 = load_nu(error_type = "MSE")[patient]

grid_name = "refine"
grid = True if grid_name == "refine" else False

#save_first_450(nu2, refined_grid=grid)
#Patient3: 504.93 ,   4.235
# p3 next to try 506.65 ,   1.7
ICD_time, ICD_duration = 800, 0#503.728,   1.843
try:
    new_signal = np.load(f'Definitive_Patients/{nu2}_{ICD_time : .3f}_{ICD_duration : .3f}_{grid_name}')
except:
    new_signal = generate_last_350(nu2, ICD_time=ICD_time, ICD_duration=ICD_duration, refined_grid=grid)
    np.save(f'Definitive_Patients/{nu2}_{ICD_time : .3f}_{ICD_duration : .3f}_{grid_name}', new_signal[0])
ori_signal = np.load("signals_3_patients.npy")

# add save
plt.title(f'Patient {nu2 : .4f}, Time {ICD_time : .3f}, Duration {ICD_duration : .3f}')
plt.plot(ori_signal[patient-1,1,:],ori_signal[patient-1,0,:],"g")
plt.plot(new_signal[0,1,:],new_signal[0,0,:],"b")
plt.legend(['New Signal', 'Original Signal'])
plt.plot([450,450],[-1.5,1],"r")
plt.show()