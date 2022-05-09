import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("scripts")
from utils2 import save_first_450, generate_last_350, load_nu, l2_int

patient1 = { 'ICD_time' : 478.945, 'ICD_duration' : 0.765 }
patient2 = { 'ICD_time' : 493.000, 'ICD_duration' : 0.805}
patient3 = { 'ICD_time' : 505.835, 'ICD_duration' : 1.925}

patients_list = [patient1, patient2, patient3]

patient = 1
nu2 = load_nu(error_type = "MSE")[patient]

grid_name = "refine"
grid = True if grid_name == "refine" else False

#save_first_450(nu2, refined_grid=grid)
ICD_time, ICD_duration = patients_list[patient-1]['ICD_time'], patients_list[patient-1]['ICD_duration']


try:
    new_signal = np.load(f'Definitive_Patients/{nu2}_{ICD_time : .3f}_{ICD_duration : .3f}_{grid_name}')
except:
    new_signal = generate_last_350(nu2, ICD_time=ICD_time, ICD_duration=ICD_duration, refined_grid=grid)

ori_signal = np.load("signals_3_patients.npy")

plt.figure()
plt.title(f'Patient {nu2 : .6f}, Time {ICD_time : .3f}, Duration {ICD_duration : .3f}')
plt.plot(ori_signal[patient-1,1,:],ori_signal[patient-1,0,:],"g")
plt.xlabel('Time [ms]')
plt.plot(new_signal[0,1,:],new_signal[0,0,:],"b")
plt.legend(['Original Signal', 'New Signal'])
plt.plot([450,450],[-1.5,1],"r")
plt.show()