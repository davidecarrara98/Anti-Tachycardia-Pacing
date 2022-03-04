# Import
import numpy as np
from BayesianOptimizer import BayesOptimizer
import functions_davide

# Set seed for reproducibility
np.random.seed(2304)
# nu2 can belong to [0.0116,0.0124]
patients = np.load('signals_3_patients.npy')

# COARSE GRID - estimated with MSE: 0.0124 - estimated with MAE: 0.0124 - estimated with comb: 0.0124
patient1 = np.array(patients[0])
bayes_opt1 = BayesOptimizer(patient1, name_patient="Patient 1", niter=10, k=3, T=450, load_all=True,
                            min_iter=0, error_function=functions_davide.l12_norm, refined_grid=True)
nu1, mod1 = bayes_opt1.optimize()

# COARSE GRID - estimated with MSE: 0.01205325 - estimated with MAE: 0.01204525 - estimated with comb: 0.01204605
patient2 = np.array(patients[1])
<<<<<<< Updated upstream
bayes_opt2 = BayesOptimizer(patient2, name_patient="Patient 2", niter=86, k=4, T=450, load_all=True,
                            min_iter=0, error_function=functions_davide.l12_norm)
=======
bayes_opt2 = BayesOptimizer(patient2, name_patient="Patient 2", niter=10, k=3, T=450, load_all=True,
                            min_iter=0, error_function=functions_davide.l12_norm, refined_grid=True)
>>>>>>> Stashed changes
nu2, mod2 = bayes_opt2.optimize()

# COARSE GRID - estimated with MSE: 0.01183063 - estimated with MAE: 0.01182663 - estimated with comb: 0.01182743
patient3 = np.array(patients[2])
<<<<<<< Updated upstream
bayes_opt3 = BayesOptimizer(patient3, name_patient="Patient 3", niter=86, k=4, T=450, load_all=True,
                            min_iter=0, error_function=functions_davide.l12_norm)
=======
bayes_opt3 = BayesOptimizer(patient3, name_patient="Patient 3", niter=10, k=3, T=450, load_all=True,
                            min_iter=0, error_function=functions_davide.l12_norm, refined_grid=True)
>>>>>>> Stashed changes
nu3, mod3 = bayes_opt3.optimize()
