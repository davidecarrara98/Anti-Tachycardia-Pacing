import matplotlib.pyplot as plt
import os
import utils1
from matplotlib import pylab
import numpy as np
import GPy

def extract_nu(name):
    no_time = name[:-8]
    nu_string = no_time[11:]
    nu_float = float(nu_string)
    return nu_float

def acquisition_function(yp, vp, beta=2):
    return - yp + beta * np.sqrt(vp)


class BayesOptimizer:

    def __init__(self, observed_patient, name_patient = "Unkown", T=450, niter=30, k=3, load_all=False,
                 error_function=utils1.l2_norm, min_iter=4, refined_grid=False):
        self.patient, self.name_patient = observed_patient, name_patient
        self.est_nu2 = None
        self.data_vec = None
        self.nu_min, self.nu_max = 0.0116, 0.0124
        self.k, self.T = k, T
        self.model, self.kernel = None, None
        self.niter, self.min_iter = niter, min_iter
        self.NU_domain, self.npoints = None, 1000
        self.load_all, self.refined_grid = load_all, refined_grid
        self.error_function, self.dimensions = error_function, {}

    def choose_dimensions(self):
        if self.refined_grid:
            self.dimensions = {'N':256, 'M':128, 'delta_t':0.005}
        else:
            self.dimensions = {'N': 128, 'M': 64, 'delta_t': 0.01}
        return


    def start_optimization(self):
        starting_nus, error_list = [], []
        if self.load_all:
            patients_dir = 'Patients/'
            filenames = next(os.walk(patients_dir), (None, None, []))[2]
            for name in filenames:
                if int(name[-7:-4]) == self.T:
                    nu_file = np.float(name[11:-8])
                    starting_nus.append(nu_file)
                    new_p = np.load(f'Patients/new_patient{nu_file}_{self.T}.npy')
                    new_error = self.error_function(new_p, self.patient)
                    error_list.append(new_error)
        self.k = max(self.k, len(error_list))

        if not self.load_all or len(error_list) == 0:
            starting_nus = np.linspace(self.nu_min, self.nu_max, self.k)
            for nu in starting_nus:
                try:
                    new_p = np.load(f'Patients/new_patient{nu}_{self.T}.npy')

                except:
                    new_p = utils1.generate_curve(T=self.T, nu2=nu, N=self.dimensions['N'],
                                                  M=self.dimensions['M'], delta_t=self.dimensions['delta_t'])[0]
                    new_p = np.array(new_p)
                    np.save(f'Patients/new_patient{nu}_{self.T}', new_p)

                new_error = self.error_function(new_p, self.patient)
                error_list.append(new_error)

        self.data_vec = np.array([([i], [j]) for i, j in zip(starting_nus, error_list)])
        print(f'Starting optimization with {self.data_vec.shape[0]} initial points')
        return

    def initialize_gp(self):
        self.kernel = GPy.kern.RBF(input_dim=1)
        self.model = GPy.models.GPRegression(self.data_vec[:, 0], self.data_vec[:, 1], self.kernel)
        self.model['rbf.lengthscale'].constrain_bounded(1e-5, 1e-4, warning=False)
        self.model.optimize(messages=False)
        if not self.load_all:
            print(self.model)
            self.model.plot()
            plt.title(f"Start search on {self.name_patient}")
            plt.xlabel("nu")
            plt.ylabel("MSE")
            pylab.show(block=True)
        return

    def optimize_gp(self):

        nu_range = self.nu_max - self.nu_min
        dnu = nu_range / self.npoints

        niter = self.niter - self.k
        niter = max(niter, self.min_iter)

        for iterations in range(niter):

            self.NU_domain = np.linspace(self.nu_min + dnu * np.random.uniform(),
                                         self.nu_max + dnu * np.random.uniform(),
                                         self.npoints).reshape(-1, 1)

            print(niter - iterations, ' iterations remaining')
            yp, vp = self.model.predict(self.NU_domain)
            evaluated_data = acquisition_function(yp, vp)

            self.est_nu2 = self.NU_domain[np.argmax(evaluated_data)]
            if self.est_nu2 < self.nu_min:
                self.est_nu2 = self.nu_min
            if self.est_nu2 > self.nu_max:
                self.est_nu2 = self.nu_max

            try:
                new_p = np.load(f'Patients/new_patient{self.est_nu2[0]}_{self.T}.npy')

            except:
                new_p = utils1.generate_curve(T=self.T, nu2=self.est_nu2, N=self.dimensions['N'],
                                              M=self.dimensions['M'], delta_t=self.dimensions['delta_t'])[0]
                new_p = np.array(new_p)
                try:
                    np.save(f'Patients/new_patient{self.est_nu2[0]}_{self.T}', new_p)
                except:
                    np.save(f'Patients/new_patient{self.est_nu2}_{self.T}', new_p)

            new_error = self.error_function(new_p, self.patient)

            try:
                self.data_vec = np.append(self.data_vec, [(self.est_nu2, [new_error])], axis=0)
            except:
                self.data_vec = np.append(self.data_vec, [([self.est_nu2], [new_error])], axis=0)

            self.model = GPy.models.GPRegression(self.data_vec[:, 0], self.data_vec[:, 1], self.kernel)
            self.model['rbf.lengthscale'].constrain_bounded(1e-5, 1e-4, warning=False)
            self.model.optimize(messages=False)

        return

    def results(self):

        print(self.model)
        self.model.plot()
        plt.title(f"End search on {self.name_patient}")
        plt.xlabel("nu")
        plt.ylabel("MSE")
        pylab.show(block=True)

        self.NU_domain = np.linspace(self.nu_min, self.nu_max,
                                     self.npoints).reshape(-1, 1)
        yp, vp = self.model.predict(self.NU_domain)
        self.est_nu2 = self.NU_domain[np.argmin(yp)]
        print('Chosen Nu is : ', self.est_nu2)

        return

    def optimize(self):
        self.choose_dimensions()
        self.start_optimization()
        self.initialize_gp()
        self.optimize_gp()
        self.results()
        return self.est_nu2, self.model

