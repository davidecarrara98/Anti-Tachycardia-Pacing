# Import
import numpy as np
from BayesianOptimizer import BayesOptimizer

#Set seed for reproducibility
np.random.seed(2304)
# nu2 can belong to [0.0116,0.0124]
patients = np.load('signals_3_patients.npy')

# patient 1
patient1 = np.array(patients[0])
bayes_opt1 = BayesOptimizer(patient1, niter=30, k=4, T=450)
nu1, mod1 = bayes_opt1.optimize()

# patient 2
patient2 = np.array(patients[1])
bayes_opt2 = BayesOptimizer(patient2, niter=30, k=4, T=450)
nu2, mod2 = bayes_opt2.optimize()

# patient 3
patient3 = np.array(patients[2])
bayes_opt3 = BayesOptimizer(patient3, niter=30, k=4, T=450)
nu3, mod3 = bayes_opt3.optimize()