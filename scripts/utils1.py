import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

def l2_norm(true_curve, noisy_curve):
    minlength = np.minimum(true_curve.shape[1], noisy_curve.shape[1])
    MSE = mean_squared_error(true_curve[0, :minlength], noisy_curve[0, :minlength])
    return MSE

def l1_norm(true_curve, noisy_curve):
    minlength = np.minimum(true_curve.shape[1], noisy_curve.shape[1])
    MAE = mean_absolute_error(true_curve[0, :minlength], noisy_curve[0, :minlength])
    return MAE

def l12_norm(true_curve, noisy_curve):
    return l1_norm(true_curve, noisy_curve) + l2_norm(true_curve, noisy_curve)

# inner function for solution representation
def plotsolution(u_k, k):
    plt.clf()

    # plt.title(f"Solution at t = {k * delta_t:.3f} ms")
    plt.xlabel("x")
    plt.ylabel("y")

    # solution plot u at time-step k
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    plt.axis('equal')

    plt.show(block=False)
    plt.pause(0.01)

    return plt


# inner functions for differentiation
def make_kernel(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1, 1])
    return tf.constant(a, dtype=1)


def simple_conv(x, k):
    """A simplified 2D convolution operation"""
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y[0, :, :, 0]


def laplace(x):
    """Compute the 2D laplacian of an array"""
    laplace_iso_k = make_kernel([[0.25, 0.5, 0.25],
                                 [0.5, -3., 0.5],
                                 [0.25, 0.5, 0.25]])
    laplace_k = make_kernel([[0.0, 1.0, 0.0],
                             [1.0, -4.0, 1.0],
                             [0.0, 1.0, 0.0]])
    return simple_conv(x, laplace_iso_k)


def laplace_fiber(x):
    """Compute the 2D laplacian of an array"""
    laplace_fib_k = make_kernel([[0.0, 1.0, 0.0],
                                 [0, -2., 0.0],
                                 [0.0, 1.0, 0.0]])
    return simple_conv(x, laplace_fib_k)


def diff_y(x):
    """Compute the 2D laplacian of an array"""
    diff_k = make_kernel([[0.0, 0.5, 0.0],
                          [0, 0.0, 0.0],
                          [0.0, -0.5, 0.0]])
    return simple_conv(x, diff_k)


def diff_x(x):
    """Compute the 2D laplacian of an array"""
    diff_k = make_kernel([[0.0, 0.0, 0.0],
                          [-0.5, 0.0, 0.5],
                          [0.0, 0.0, 0.0]])
    return simple_conv(x, diff_k)


def generate_curve(T, nu2, ICD_time=460, ICD_duration=5, N=128, M=64, delta_t=0.01):
    # space discretization
    N = np.int32(N)
    M = np.int32(M)
    h = 2 / N

    # time discretization
    delta_t = tf.constant(delta_t, dtype=tf.float32, shape=())
    max_iter_time = np.int32(T / delta_t) + 1
    scaling_factor = np.int32(1.0 / delta_t)

    Trigger = 322  # [ms]
    S2 = Trigger / delta_t  # *4
    num_sim = 1
    save_flag = True

    # initialization
    signals = np.zeros([num_sim, 3, np.int32(max_iter_time / scaling_factor) + 1], dtype=np.float32)

    # timing of shock
    ICD_time = np.int32(ICD_time / delta_t)
    # duration of the shock
    ICD_duration = ICD_duration
    # amplitude of the shock
    ICD_amplitude = 100.0 #unused

    # Initial Condition
    ut_init = np.zeros([N, M], dtype=np.float32)
    Iapp_IC = np.zeros([N, M], dtype=np.float32)
    Iapp_init = np.zeros([N, M], dtype=np.float32)
    Iapp_ICD = np.zeros([N, M], dtype=np.float32)
    r_coeff = 1.2 + np.zeros([N, M], dtype=np.float32)

    distance_matrix_1 = np.zeros([N, M], dtype=np.float32)  # 0.05+1.95*np.random.rand(N, N)
    distance_matrix_2 = np.zeros([N, M], dtype=np.float32)  # 0.05+1.95*np.random.rand(N, N)
    for i in range(N):
        for j in range(M):
            distance_matrix_1[i, j] = 1 / (np.sqrt((i * h - 1) ** 2 + (j * h - 1.25) ** 2))
            distance_matrix_2[i, j] = 1 / (np.sqrt((i * h - 1) ** 2 + (j * h + 0.25) ** 2))

    # square input
    Iapp_init[np.int32(N / 2 - 0.4 / h):np.int32(N / 2 + 0.4 / h),
    np.int32(M / 2 - 0.15 / h):np.int32(M / 2 + 0.15 / h)] = 1.0
    Iapp_ICD[0:np.int32(0.25 / h), np.int32(0.5 / h):np.int32(0.65 / h)] = 1.0
    Iapp_ICD[N - np.int32(0.25 / h):N - 1, np.int32(0.5 / h):np.int32(0.65 / h)] = 1.0

    # side
    Iapp_IC[:, 0:np.int32(0.05 / h)] = 100.0
    Ulist = []

    # physical coefficients
    nu_0 = tf.constant(1.5, dtype=tf.float32, shape=())
    nu_1 = tf.constant(4.4, dtype=tf.float32, shape=())

    # parameter to be modified in the interval [0.0116,0.0124]
    nu_2 = tf.constant(nu2, dtype=tf.float32, shape=())

    nu_3 = tf.constant(1.0, dtype=tf.float32, shape=())
    v_th = tf.constant(13, dtype=tf.float32, shape=())
    v_pk = tf.constant(100, dtype=tf.float32, shape=())
    D_1 = tf.constant(0.003 / (h ** 2), dtype=tf.float32, shape=())
    D_2 = tf.constant(0.000315 / (h ** 2), dtype=tf.float32, shape=())

    # Create variables for simulation
    Ut = tf.Variable(ut_init)
    Wt = tf.Variable(0 * ut_init)
    Iapp = tf.Variable(Iapp_init)
    IappICD = tf.Variable(Iapp_ICD)
    IappIC = tf.Variable(Iapp_IC)
    Dr = tf.Variable(r_coeff, dtype=np.float32)

    for i in tqdm(range(max_iter_time), desc=f'Building Curve - Using nu2 = {nu_2}', leave=False):

        # sinus rhythm
        if ((i > -1) & (i < 1 + np.int32(2 / delta_t))) | \
                ((i > np.int32(200 / delta_t)) & (i < np.int32(202 / delta_t))):
            coeff_init = 10.0
        else:
            coeff_init = 0.0

        # extra-stim
        if (i > S2) & (i < S2 + np.int32(2 / delta_t)):
            coeff = 100.0
        else:
            coeff = 0.0

        # ATP impulse
        if (i > ICD_time) & (i < ICD_time + np.int32(ICD_duration / delta_t)):
            coeff_ICD = 100
        else:
            coeff_ICD = 0.0

        # nonlinear terms
        I_ion = nu_0 * Ut * (1.0 - Ut / v_th) * (1.0 - Ut / v_pk) + nu_1 * Wt * Ut
        g_ion = nu_2 * (Ut / v_pk - nu_3 * Wt)

        # update the solution
        Ut = Ut + delta_t * (Dr * D_2 * laplace(Ut) + Dr * (D_1 - D_2) * laplace_fiber(Ut)
                             - I_ion + coeff_init * IappIC + coeff * Iapp + coeff_ICD * IappICD)
        Wt = Wt + delta_t * g_ion

        # ghost nodes
        tmp_u = Ut.numpy()

        tmp_u[0, :] = tmp_u[2, :]
        tmp_u[N - 1, :] = tmp_u[N - 3, :]
        tmp_u[:, 0] = tmp_u[:, 2]
        tmp_u[:, M - 1] = tmp_u[:, M - 3]

        Ut = tf.Variable(tmp_u)

        Ulist.append(Ut)

    if save_flag:

        for i in tqdm(range(max_iter_time + 1), desc=f'Compiling Curve - Using nu2 = {nu_2}', leave=False):
            k = np.int32(i / scaling_factor)
            if (np.mod(i, scaling_factor) == 0):
                ref = Ulist[i][np.int32(N / 2)][np.int32(M / 2)]

                # pseudo ECG
                signals[0, 0, k] = 1 / (h ** 2) * np.sum(
                    diff_x(Ulist[i][:][:]) * diff_y(distance_matrix_1) + diff_y(Ulist[i][:][:]) * diff_y(
                        distance_matrix_1)) \
                                   - 1 / (h ** 2) * np.sum(
                    diff_x(Ulist[i][:][:]) * diff_y(distance_matrix_2) + diff_y(Ulist[i][:][:]) * diff_y(
                        distance_matrix_2))

                signals[0, 1, k] = i * delta_t

                # ICD trace
                if (i > ICD_time) & (i < ICD_time + np.int32(ICD_duration / delta_t)):
                    signals[0, 2, k] = ICD_amplitude
                else:
                    signals[0, 2, k] = 0.0

        signals[0, 0, :] = signals[0, 0, :] / np.amax(signals[0, 0, :])

    return signals
