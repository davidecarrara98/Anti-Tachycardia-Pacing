import os
import GPy
import sys
import numpy as np
from matplotlib import pylab
import  matplotlib.pyplot as plt
from utils2 import generate_last_350
from BayesianOptimizer2D import BayesOptimizer2D
from BayesianOptimizer import acquisition_function

sys.path.append("scripts")
from utils2 import load_nu, custom_loss

def boosting_error(curve, duration):
    integral = np.linalg.norm(curve[0, 700:]) / 100
    if integral > 10e-10:
        return 2.6
    else:
        return duration

class BayesBooster:
    def __init__(self, patient, k = 5, niter = 5, min_iter = 0, alpha_max=100):
        self.patient = patient
        self.nu = load_nu(error_type = "MSE")[patient]
        self.optimizer, self.load_all = None, True
        self.t, self.dur, self.mod = None, None, None
        self.alpha_min, self.alpha_max = 0, alpha_max
        self.model, self.kernel = None, None
        self.alpha, self.npoints = None, 1000
        self.grid, self.k, self.niter, self.min_iter = 'refine', k, niter, min_iter
        self.refine_grid, self.data_vec = True, None
        self.est_alpha, self.alpha_domain = None, None

    def start_optimization(self):
        starting_alphas, error_list = [], []
        if self.load_all:
            patients_dir = f'Boosting_{self.patient}/'
            filenames = next(os.walk(patients_dir), (None, None, []))[2]
            for name in filenames:
                spl = name.split('_')
                spl = [l.strip() for l in spl]
                alpha_, time, duration = np.float(spl[0]), np.float(spl[1]), np.float(spl[2])
                try:
                    new_p = np.load(f'Boosting_{self.patient}/{alpha_}_{time : .3f}_{duration : .3f}_{self.grid}.npy')
                except FileNotFoundError:
                    new_p = np.load(f'Boosting_{self.patient}/{alpha_}_{time : .2f}_{duration : .2f}_{self.grid}.npy')
                if len(new_p.shape) == 3:
                    continue

                starting_alphas.append(alpha_)
                new_error = boosting_error(new_p, duration)
                error_list.append(new_error)


        self.data_vec = np.array([([i], [j]) for i, j in zip(starting_alphas, error_list)])
        print(f'Starting Boosting with {self.data_vec.shape[0]} initial points')

        self.k = max(self.k, len(error_list))
        return

    def initialize_gp(self):

        self.kernel = GPy.kern.RBF(input_dim=1)
        self.model = GPy.models.GPRegression(self.data_vec[:, 0], self.data_vec[:, 1], self.kernel)
        self.model['rbf.lengthscale'].constrain_bounded(1e-5, 10, warning=False)
        self.model.optimize(messages=False)
        print(self.model)

        return

    def optimize_gp(self):

        alpha_range = self.alpha_max - self.alpha_min
        dalpha = alpha_range / self.npoints

        niter = self.niter - self.k
        niter = max(niter, self.min_iter)

        for iterations in range(niter):

            self.alpha_domain = np.linspace(self.alpha_min + dalpha * np.random.uniform(),
                                            self.alpha_max + dalpha * np.random.uniform(),
                                            self.npoints).reshape(-1, 1)

            yp, vp = self.model.predict(self.alpha_domain)
            evaluated_data = acquisition_function(yp, vp)

            self.est_alpha = self.alpha_domain[np.argmax(evaluated_data)]
            if self.est_alpha < self.alpha_min:
                self.est_alpha = self.alpha_min
            if self.est_alpha > self.alpha_max:
                self.est_alpha = self.alpha_max

            print(niter - iterations, ' boosting iterations remaining')
            print(f'Evaluating alpha = {self.est_alpha}')

            self.optimizer = BayesOptimizer2D(nu=self.nu, refined_grid=True, niter=0, load_all=True, min_iter=0,
                                   error_function=custom_loss, parameter=self.est_alpha)

            t, dur, mod = self.optimizer.optimize()
            new_p = generate_last_350(nu2=self.nu, refined_grid=self.refine_grid,
                                              ICD_time=t, ICD_duration=dur)[0]
            new_p = np.array(new_p)
            try:
                np.save(f'Boosting_{self.patient}/{self.est_alpha[0]}_{t : .2f}_{dur : .2f}_{self.grid}.npy', new_p)
            except:
                np.save(f'Boosting_{self.patient}/{self.est_alpha}_{t : .2f}_{dur : .2f}_{self.grid}.npy', new_p)

            new_error = boosting_error(new_p, duration=dur)

            try:
                self.data_vec = np.append(self.data_vec, [(self.est_alpha, [new_error])], axis=0)
            except:
                self.data_vec = np.append(self.data_vec, [([self.est_alpha], [new_error])], axis=0)

            self.model = GPy.models.GPRegression(self.data_vec[:, 0], self.data_vec[:, 1], self.kernel)
            self.model['rbf.lengthscale'].constrain_bounded(1e-5, 10, warning=False)
            self.model.optimize(messages=False)

        return

    def results(self):

        print(self.model)
        self.model.plot()
        plt.title(f"End search on {self.nu}")
        plt.xlabel("alpha")
        plt.ylabel("Boosting Error")
        pylab.show(block=True)

        self.alpha_domain = np.linspace(self.alpha_min, self.alpha_max,
                                     self.npoints).reshape(-1, 1)
        yp, vp = self.model.predict(self.alpha_domain)
        self.est_alpha = self.alpha_domain[np.argmin(yp)]

        self.optimizer = BayesOptimizer2D(nu=self.nu, refined_grid=True, niter=0, load_all=True, min_iter=0,
                                          error_function=custom_loss, parameter=self.est_alpha)
        self.t, self.dur, _ = self.optimizer.optimize()

        print('Chosen Alpha is : ', self.est_alpha)

        return

    def optimize(self):
        self.start_optimization()
        self.initialize_gp()
        self.optimize_gp()
        self.results()
        return self.est_alpha, self.model

Booster = BayesBooster(patient=3, k = 0, niter = 0, min_iter = 0, alpha_max=50)
alpha, m = Booster.optimize()