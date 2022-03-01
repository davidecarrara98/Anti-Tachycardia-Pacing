"""!
@file example.py

@brief import data.

@author Stefano Pagani <stefano.pagani@polimi.it>.

@date 2022

@section Course: Scientific computing tools for advanced mathematical modelling.
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt


dataset_p = np.load('signals_3_patients.npy')

for ind_s in range(3):
    plt.plot(dataset_p[ind_s,1][:],dataset_p[ind_s,0][:])

plt.show()
