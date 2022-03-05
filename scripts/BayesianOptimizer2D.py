import matplotlib.pyplot as plt
import os

from matplotlib import pylab
import numpy as np
import GPy
from utils2 import l2_int, generate_last_350

def acquisition_function(yp, vp, beta=2):
    return - yp + beta * np.sqrt(vp)

def cartesian_product(x, y):
    return np.transpose([np.tile(x, len(y)),
                            np.repeat(y, len(x))])


class BayesOptimizer2D:

    def __init__(self, nu, name_patient = "Unkown", niter=30, k=4, load_all=False,
                 error_function=l2_int, min_iter=4, refined_grid=False):
        self.nu, self.name_patient = nu, name_patient
        self.X, self.Y = None, None
        self.est_t, self.est_dur = None, None
        self.t_min, self.t_max = 450, 525
        self.k = k
        self.model, self.kernel = None, None
        self.niter, self.min_iter = niter, min_iter
        self.prediction_grid = None
        self.load_all, self.refined_grid = load_all, refined_grid
        self.error_function = error_function
        self.grid = 'refine' if self.refined_grid == True else 'coarse'
        self.dur_min = 0.005 if self.refined_grid == True else 0.01
        self.dur_max = 10


    def start_optimization(self):
        starting_time, starting_duration, error_list = [], [], []
        starting_values = []
        if self.load_all:
            pass
        self.k = max(self.k, len(error_list))

        if not self.load_all or len(error_list) == 0:
            starting_time = np.linspace(self.t_min, self.t_max, int(self.k/2))
            starting_duration = np.array([2.5, 7.5])
            starting_values = cartesian_product(starting_time, starting_duration)

            for values in starting_values:
                if values[0] + values[1] > self.t_max : values[0] = self.t_max - values[1]
                time, duration = values[0], values[1]
                try:
                    new_p = np.load(f'Definitive_Patients/{self.nu}_{time : .2f}_{duration : .2f}_{self.grid}.npy')

                except:
                    new_p = generate_last_350(nu2=self.nu, refined_grid=self.refined_grid,
                                              ICD_time=time, ICD_duration=duration)[0]
                    new_p = np.array(new_p)
                    np.save(f'Definitive_Patients/{self.nu}_{time : .2f}_{duration : .2f}_{self.grid}', new_p)

                new_error = self.error_function(new_p)
                error_list.append(new_error)

        self.X = np.array([i for i in starting_values])
        self.Y = np.array([[j] for j in error_list])
        print(f'Starting optimization with {self.X.shape[0]} initial points')
        return

    def initialize_gp(self):
        self.kernel = GPy.kern.RBF(input_dim=2)
        self.model = GPy.models.GPRegression(self.X, self.Y, self.kernel)
        self.model['rbf.lengthscale'].constrain_bounded(1, 20, warning=False)
        self.model.optimize(messages=False)
        if not self.load_all:
            print(self.model)
            self.model.plot()
            plt.title(f"Start search on nu = {self.nu : .6f}")
            plt.xlabel("Activation Time")
            plt.ylabel("Duration")
            pylab.show(block=True)
        return

    def optimize_gp(self):

        #nu_range = self.nu_max - self.nu_min
        #dnu = nu_range / self.npoints

        niter = self.niter - self.k
        niter = max(niter, self.min_iter)

        for iterations in range(niter):

            self.prediction_grid = np.mgrid[self.t_min:self.t_max:self.dur_min,
                              self.dur_min:self.dur_max:self.dur_min].reshape(2,-1).T

            print(niter - iterations, ' iterations remaining')
            yp, vp = self.model.predict(self.prediction_grid)
            evaluated_data = acquisition_function(yp, vp)

            self.est_t, self.est_dur = self.prediction_grid[np.argmax(evaluated_data)]
            if self.est_t < self.t_min:
                self.est_t = [self.t_min]
            if self.est_t > self.t_max:
                self.est_t = [self.t_max]
            if self.est_dur < self.dur_min:
                self.est_dur = [self.dur_min]
            if self.est_dur > self.dur_max:
                self.est_dur = [self.dur_max]

            try:
                new_p = np.load(f'Definitive_Patients/{self.nu}_{self.est_t : .2f}_{self.est_dur : .2f}_{self.grid}.npy')

            except:
                new_p = generate_last_350(nu2=self.nu, refined_grid=self.refined_grid,
                                              ICD_time=self.est_t, ICD_duration=self.est_dur)[0]
                new_p = np.array(new_p)
                np.save(f'Definitive_Patients/{self.nu}_{self.est_t[0] : .2f}_{self.est_dur[0] : .2f}_{self.grid}.npy', new_p)
                #except:
                #    np.save(f'Patients/new_patient{self.est_nu2}_{self.T}', new_p)

            new_error = self.error_function(new_p)

            #try:
            self.X = np.append(self.X, [self.est_t[0], self.est_dur[0]], axis=0)
            self.Y = np.append(self.Y, [new_error], axis=0)
            #except:
            #    self.data_vec = np.append(self.data_vec, [([self.est_nu2], [new_error])], axis=0)

            self.model = GPy.models.GPRegression(self.X, self.Y, self.kernel)
            self.model['rbf.lengthscale'].constrain_bounded(1, 20, warning=False)
            self.model.optimize(messages=False)

        return

    def results(self):

        print(self.model)
        self.model.plot()
        plt.title(f"End search on nu = {self.nu : .6f}")
        plt.xlabel("Activation Time")
        plt.ylabel("Duration")
        pylab.show(block=True)

        self.prediction_grid = np.mgrid[self.t_min:self.t_max:self.dur_min,
                               self.dur_min:self.dur_max:self.dur_min].reshape(2, -1).T
        yp, vp = self.model.predict(self.prediction_grid)

        mask = np.ones(shape=self.prediction_grid.shape[0])
        for ind, values in enumerate(self.prediction_grid):
            if values[0] + values[1] > self.t_max:
                mask[ind] = np.inf

        final_pred = np.multiply(yp.squeeze(), mask)
        self.est_t, self.est_dur = self.prediction_grid[np.argmin(final_pred)]
        print(f'Chosen Activation Time is : {self.est_t : .6f}')
        print(f'Chosen Duration is : {self.est_dur: .6f}')

        return

    def optimize(self):
        self.start_optimization()
        self.initialize_gp()
        self.optimize_gp()
        self.results()
        return self.est_t, self.est_dur, self.model