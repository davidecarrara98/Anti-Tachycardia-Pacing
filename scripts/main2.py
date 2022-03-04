import json
import numpy as np
import utils2

patient_name = 'patient1'
nu2 = 0.0124
grid = 'coarse'
utils2.save_first_450(nu2=0.0124, patient_name='patient1', refined_grid=False)

# Opening JSON file
f = open(f'../First_450/{patient_name}_{nu2}_{grid}.json')

# returns JSON object as
# a dictionary
data = json.load(f)
data = np.asarray(data["a"])
print(data)