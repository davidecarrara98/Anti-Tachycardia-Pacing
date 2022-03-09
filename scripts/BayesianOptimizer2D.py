import matplotlib.pyplot as plt
import os

from matplotlib import pylab
import numpy as np
import GPy
from utils2 import l2_int, generate_last_350, custom_loss

def acquisition_function(yp, vp, beta=2, length=0):
    focus = True if length > 60 else False
    if focus:
        beta = 0.1
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
        self.prediction_grid, self.final_pred = None, None
        self.load_all, self.refined_grid = load_all, refined_grid
        self.error_function = error_function
        self.grid = 'refine' if self.refined_grid == True else 'coarse'
        self.dur_min = 0.005 if self.refined_grid == True else 0.01
        self.dur_max = 10


    def start_optimization(self):
        starting_time, starting_duration, error_list = [], [], []
        starting_values = []
        if self.load_all:
            patients_dir = 'Definitive_Patients/'
            filenames = next(os.walk(patients_dir), (None, None, []))[2]
            for name in filenames:
                spl = name.split('_')
                spl = [l.strip() for l in spl]
                if np.abs(np.float(spl[0]) - self.nu) < 10e-9 and spl[3][:-4] == self.grid :
                    time, duration = np.float(spl[1]), np.float(spl[2])
                    try:
                        new_p = np.load(f'Definitive_Patients/{self.nu}_{time : .3f}_{duration : .3f}_{self.grid}.npy')
                    except FileNotFoundError:
                        new_p = np.load(f'Definitive_Patients/{self.nu}_{time : .2f}_{duration : .2f}_{self.grid}.npy')
                    if len(new_p.shape) == 3:
                        continue

                    starting_time.append(time)
                    starting_duration.append(duration)
                    new_error = self.error_function(new_p, duration)
                    error_list.append(new_error)
            self.X = np.array([i for i in zip(starting_time, starting_duration)])
            self.Y = np.array([[j] for j in error_list])
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

                new_error = self.error_function(new_p, duration)
                error_list.append(new_error)

            self.X = np.array([i for i in starting_values])
            self.Y = np.array([[j] for j in error_list])
        print(f'Starting optimization with {self.X.shape[0]} initial points')
        return

    def initialize_gp(self):
        self.kernel = GPy.kern.RBF(input_dim=2)
        Xtr = self.X / np.array([10, 1])
        self.model = GPy.models.GPRegression(Xtr, self.Y, self.kernel)
        self.model['rbf.lengthscale'].constrain_bounded(1, 100, warning=False)
        self.model.optimize(messages=False)
        if not False:
            print(self.model)
            self.model.plot()
            plt.title(f"Start search on nu = {self.nu : .6f}")
            plt.xlabel("Activation Time / 10")
            plt.ylabel("Duration")
            pylab.show(block=True)
        return

    def optimize_gp(self):

        #nu_range = self.nu_max - self.nu_min
        #dnu = nu_range / self.npoints

        niter = self.niter - self.k
        niter = max(niter, self.min_iter)

        for iterations in range(niter):
            print(niter - iterations, ' iterations remaining')

            self.prediction_grid = np.mgrid[self.t_min:self.t_max:self.dur_min,
                                   self.dur_min:self.dur_max:self.dur_min].reshape(2, -1).T
            length = self.prediction_grid.shape[0]
            predict_list = [self.prediction_grid[np.int32(i * length / 20):np.int32( (i+1) *length / 20), :]
                            for i in range(20)]
            yp, vp = np.empty(shape=(0, 1)), np.empty(shape=(0, 1))
            for l in predict_list:
                y, v = self.model.predict(l / np.array([10, 1]))
                yp, vp = np.concatenate([yp, y]), np.concatenate([vp, v])

            evaluated_data = acquisition_function(yp, vp, self.model.X.shape[0])
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
                try:
                    np.save(f'Definitive_Patients/{self.nu}_{self.est_t[0] : .2f}_{self.est_dur[0] : .2f}_{self.grid}.npy', new_p)
                except:
                    np.save(f'Definitive_Patients/{self.nu}_{self.est_t : .2f}_{self.est_dur : .2f}_{self.grid}.npy', new_p)

            if len(new_p.shape) == 3:
                continue
            new_error = self.error_function(new_p, self.est_dur)

            #try:
            self.X = np.append(self.X, [[self.est_t, self.est_dur]], axis=0)
            self.Y = np.append(self.Y, [[new_error]], axis=0)
            #except:
            #    self.data_vec = np.append(self.data_vec, [([self.est_nu2], [new_error])], axis=0)

            Xtr = self.X / np.array([10, 1])
            self.model = GPy.models.GPRegression(Xtr, self.Y, self.kernel)
            self.model['rbf.lengthscale'].constrain_bounded(1, 100, warning=False)
            self.model.optimize(messages=False)

        return

    def results(self):

        print(self.model)
        self.model.plot()

        self.prediction_grid = np.mgrid[self.t_min:self.t_max:self.dur_min,
                               self.dur_min:self.dur_max:self.dur_min].reshape(2, -1).T
        length = self.prediction_grid.shape[0]
        predict_list = [self.prediction_grid[np.int32(i * length / 20):np.int32((i + 1) * length / 20), :]
                        for i in range(20)]
        yp = np.empty(shape = (0,1))
        for l in predict_list:
            y, v = self.model.predict( l / np.array([10, 1]))
            yp = np.concatenate([yp, y])


        plt.title(f"End search on nu = {self.nu : .6f}")
        plt.xlabel("Activation Time / 10")
        plt.ylabel("Duration")
        pylab.show(block=True)


        mask = np.ones(shape=self.prediction_grid.shape[0])
        for ind, values in enumerate(self.prediction_grid):
            if values[0] + values[1] > self.t_max:
                mask[ind] = np.inf

        self.final_pred = np.multiply(yp.squeeze(), mask)
        self.est_t, self.est_dur = self.prediction_grid[np.argmin(self.final_pred)]
        print(f'Chosen Activation Time is : {self.est_t : .6f}')
        print(f'Chosen Duration is : {self.est_dur: .6f}')

        return

    def plot_3d(self):
        t_vec = np.arange(self.t_min, self.t_max, self.dur_min * 10)
        d_vec = np.arange(self.dur_min, self.dur_max, self.dur_min * 10)
        t_vec, d_vec = np.meshgrid(t_vec, d_vec)
        prediction_grid = np.mgrid[self.t_min:self.t_max:self.dur_min * 10, self.dur_min:self.dur_max:self.dur_min * 10].reshape(2, -1).T
        yp, vp = self.model.predict(prediction_grid / np.array([10, 1]))

        al = yp.reshape(1500, 200).T
        plt.figure(figsize=(20, 20), dpi=64)
        ax = plt.axes(projection='3d')
        ax.view_init(elev=20, azim=-40)
        ax.plot_surface(t_vec, d_vec, al, rstride=1, cstride=1, cmap='seismic',
                        antialiased=False, edgecolor='none')
        ax.set_title(f'MSE evaluated for nu = {self.nu : .6f}', fontsize=60)
        ax.set_xlabel('Activation Time', fontsize=35, labelpad=30)
        ax.set_ylabel('Duration', fontsize=35, labelpad=30)
        ax.set_zlabel('MSE', fontsize=35, labelpad=30)
        # Set tick font size
        for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            label.set_fontsize(30)

        mse = self.model.predict(np.array([[self.est_t/10, self.est_dur]]))
        plt.show()

    def optimize(self):
        self.start_optimization()
        self.initialize_gp()
        self.optimize_gp()
        self.results()
        self.plot_3d()
        return self.est_t, self.est_dur, self.model