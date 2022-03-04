import numpy as np
import utils2
import matplotlib.pyplot as plt

nu2 = 0.012400
grid = False #True is fine
#utils2.save_first_450(nu2, refined_grid=grid)
signal = utils2.generate_last_350(nu2, ICD_time=500, ICD_duration=10, refined_grid=grid)
plt.plot(signal[0,1,:],signal[0,0,:], "r")