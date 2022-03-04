import json
from json import JSONEncoder
import numpy as np
import utils2
a = np.array([[5], [3]])
d = {'a':a}
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

with open(f'../First_450/a.json', 'w') as fp:
    json.dump(d, fp, cls=NumpyArrayEncoder)

#utils2.save_first_450(nu2=0.0124, patient_name='patient1', refined_grid=False)

# Opening JSON file
f = open(f'../First_450/a.json')

# returns JSON object as
# a dictionary
data = json.load(f)
data = np.asarray(data["a"])
print(data)