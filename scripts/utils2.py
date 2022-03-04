from utils1 import *
from json import JSONEncoder
import json

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def save_first_450(nu2, ICD_time=460, ICD_duration=5,refined_grid=True, patient_name='patient'):
    # space discretization
    T = 45
    if refined_grid:
        N = np.int32(256)
        M = np.int32(128)
        delta_t = 0.005
    else:
        N = np.int32(128)
        M = np.int32(64)
        delta_t = 0.01

    h = 2 / N

    # time discretization
    delta_t = tf.constant(delta_t, dtype=tf.float32, shape=())
    max_iter_time = np.int32(T / delta_t) + 1
    scaling_factor = np.int32(1.0 / delta_t)

    Trigger = 322  # [ms]
    S2 = Trigger / delta_t  # *4
    num_sim = 1

    # initialization
    #signals = np.zeros([num_sim, 3, np.int32(max_iter_time / scaling_factor) + 1], dtype=np.float32)

    # timing of shock
    #ICD_time = np.int32(ICD_time / delta_t)
    # duration of the shock
    #ICD_duration = ICD_duration
    # amplitude of the shock
    #ICD_amplitude = 1.0

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

    for i in tqdm(range(max_iter_time), desc=f'Building Curve - Using nu2 = {nu_2 : .4f}', leave=False):

        # sinus rhythm
        if ((i > -1) & (i < 1 + np.int32(2 / delta_t))) | \
                ((i > np.int32(200 / delta_t)) & (i < np.int32(202 / delta_t))):
            # coeff_init = ((128/N)**2)*10.0*0.02/delta_t
            coeff_init = 10.0
        else:
            coeff_init = 0.0

        # extra-stim
        if (i > S2) & (i < S2 + np.int32(2 / delta_t)):
            # coeff = ((128/N)**2)*100.0*0.02/delta_t
            coeff = 100.0
        else:
            coeff = 0.0

        # ATP impulse
        if (i > ICD_time) & (i < ICD_time + np.int32(ICD_duration / delta_t)):
            # coeff_ICD = ICD_amplitude*((128/N)**2)*100.0*0.02/delta_t
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

     #U_save = Ulist[end].numpy() ??       possiamo fare test con meno di 450 iterazioni, non so perché ne usiamo così tante per testare
    save_dict = {'Ut' : Ut.numpy(), 'Wt' :  np.array(Wt).tolist(), 'nu_2' : nu_2}
    if refined_grid: grid = 'fine'
    else: grid = 'coarse'
    with open(f'../First_450/{patient_name}_{nu_2}_{grid}.json', 'w') as fp:
        json.dump(save_dict, fp, cls=NumpyArrayEncoder)


    return