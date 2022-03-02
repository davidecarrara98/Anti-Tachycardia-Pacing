import matplotlib.pyplot as plt

import functions_davide
from matplotlib import pylab
import numpy as np
import GPy


def acquisition_function(yp, vp, beta=2):
    return - yp + beta * np.sqrt(vp)


class BayesOptimizer:
    def __init__(self, observed_patient, T=450, niter=30, k=3):
        self.patient = observed_patient
        self.est_nu2 = None
        self.data_vec = None
        self.nu_min, self.nu_max = 0.0116, 0.0124
        self.k, self.T = k, T
        self.model, self.kernel = None, None
        self.niter = niter
        self.NU_domain = None
        self.npoints = 1000

    def start_optimization(self):
        starting_nus = np.linspace(self.nu_min, self.nu_max, self.k)
        new_p_list, mse_list = [], []
        for nu in starting_nus:
            try:
                new_p = np.load(f'new_patient{nu}_{self.T}.npy')

            except:
                new_p = functions_davide.generate_curve(T=self.T, nu2=nu)[0]
                new_p = np.array(new_p)
                np.save(f'new_patient{nu}_{self.T}', new_p)

            new_mse = functions_davide.l2_norm(new_p, self.patient)
            # new_p_list.append(new_p)
            mse_list.append(new_mse)

        self.data_vec = np.array([([i], [j]) for i, j in zip(starting_nus, mse_list)])
        return

    def initialize_gp(self):
        self.kernel = GPy.kern.RBF(input_dim=1)
        self.model = GPy.models.GPRegression(self.data_vec[:, 0], self.data_vec[:, 1], self.kernel)
        self.model['rbf.lengthscale'].constrain_bounded(1e-5, 1e-4, warning=False)
        self.model.optimize(messages=False)
        print(self.model)
        self.model.plot()
        return

    def optimize_gp(self):

        nu_range = self.nu_max - self.nu_min
        dnu = nu_range / self.npoints

        niter = self.niter - self.k

        for iter in range(niter):

            self.NU_domain = np.linspace(self.nu_min + dnu * np.random.uniform(),
                                         self.nu_max + dnu * np.random.uniform(),
                                         self.npoints).reshape(-1, 1)

            print(niter - iter, ' iterations remaining')
            yp, vp = self.model.predict(self.NU_domain)
            evaluated_data = acquisition_function(yp, vp)

            self.est_nu2 = self.NU_domain[np.argmax(evaluated_data)]
            if self.est_nu2 < self.nu_min:
                self.est_nu2 = self.nu_min
            if self.est_nu2 > self.nu_max:
                self.est_nu2 = self.nu_max

            try:
                new_p = np.load(f'new_patient{self.est_nu2[0]}_{self.T}.npy')

            except:
                new_p = functions_davide.generate_curve(T=self.T, nu2=self.est_nu2)[0]
                new_p = np.array(new_p)
                try:
                    np.save(f'new_patient{self.est_nu2[0]}_{self.T}', new_p)
                except:
                    np.save(f'new_patient{self.est_nu2}_{self.T}', new_p)

            new_mse = functions_davide.l2_norm(new_p, self.patient)

            try:
                self.data_vec = np.append(self.data_vec, [(self.est_nu2, [new_mse])], axis=0)
            except:
                self.data_vec = np.append(self.data_vec, [([self.est_nu2], [new_mse])], axis=0)

            self.model = GPy.models.GPRegression(self.data_vec[:, 0], self.data_vec[:, 1], self.kernel)
            self.model['rbf.lengthscale'].constrain_bounded(1e-5, 1e-4, warning=False)
            self.model.optimize(messages=False)

        return

    def results(self):

        print(self.model)
        plt.title('MSE over nu')
        self.model.plot()
        pylab.show(block=True)

        self.NU_domain = np.linspace(self.nu_min, self.nu_max,
                                     self.npoints).reshape(-1, 1)
        yp, vp = self.model.predict(self.NU_domain)
        self.est_nu2 = self.NU_domain[np.argmin(yp)]
        print('Chosen Nu is : ', self.est_nu2)

        return

    def optimize(self):
        self.start_optimization()
        self.initialize_gp()
        self.optimize_gp()
        self.results()
        return self.est_nu2, self.model
