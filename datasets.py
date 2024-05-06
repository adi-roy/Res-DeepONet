from solver import solve_BHT
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, config
from functools import partial
from torch.utils import data
from scipy.interpolate import griddata



# Sample one training dataset
def generate_one_training_data(key, P, Q):
    Nx = 100
    Nt = 100
    mag_scale = 100
    # Numerical solution
    (x, t, UU), (u, y, s) = solve_BHT(key, Nx, Nt, P, mag_scale=100, length_scale=0.1, on_off=1)

    # Geneate subkeys
    subkeys = jr.split(key, 4)

    # Sample points from the boundary and the inital conditions
    # Here we regard the initial condition as a special type of boundary conditions
    x_bc1 = jnp.zeros((P // 3, 1))
    x_bc2 = jnp.ones((P // 3, 1))
    x_bc3 = jr.uniform(key = subkeys[0], shape = (P // 3, 1))
    x_bcs = jnp.vstack((x_bc1, x_bc2, x_bc3))

    t_bc1 = jr.uniform(key = subkeys[1], shape = (P//3 * 2, 1))
    t_bc2 = jnp.zeros((P//3, 1))
    t_bcs = jnp.vstack([t_bc1, t_bc2])

    # Training data for BC and IC
    u_train = jnp.tile(u, (P,1)) / mag_scale
    y_train = jnp.hstack([x_bcs, t_bcs])
    s_train = jnp.zeros((P, 1))
    s_train = s_train.at[:100].set(25)
    s_train = s_train.at[100:200].set(37)
    s_train = s_train.at[200:].set(37)

    # Sample collocation points
    x_r_idx= jr.choice(subkeys[2], jnp.arange(Nx), shape = (Q,1))
    x_r = x[x_r_idx]
    t_r = jr.uniform(subkeys[3], minval = 0, maxval = 1, shape = (Q,1))

    # Training data for the PDE residual
    '''For the operator'''
    u_r_train = jnp.tile(u, (Q,1)) / mag_scale
    y_r_train = jnp.hstack([x_r, t_r])
    '''For the function'''
    f_r_train = u[x_r_idx] / mag_scale
    return u_train, y_train, s_train, u_r_train, y_r_train, f_r_train



# Sample one testing dataset
def generate_one_test_data(key, P, mag_scale):
    Nx = P
    Nt = P
    (x, t, UU), (u, y, s) = solve_BHT(key, Nx , Nt, P, mag_scale, length_scale=0.1, on_off=1)

    XX, TT = jnp.meshgrid(x, t)

    u_test = jnp.tile(u, (Nx*Nt,1)) / 100
    y_test = jnp.hstack([XX.flatten()[:,None], TT.flatten()[:,None]])
    s_test = UU.T.flatten()

    return u_test, y_test, s_test


# Generate training data
def generate_training_data(key, P_train, Q_train, N):
    keys = jr.split(key, N)
    config.update("jax_enable_x64", True)
    u_train, y_train, s_train, u_r_train, y_r_train, f_r_train = vmap(generate_one_training_data, (0, None, None))(keys, P_train, Q_train)

    u_bcs_train = jnp.float32(u_train.reshape(N * P_train,-1))
    y_bcs_train = jnp.float32(y_train.reshape(N * P_train,-1))
    s_bcs_train = jnp.float32(s_train.reshape(N * P_train,-1))

    u_res_train = jnp.float32(u_r_train.reshape(N * Q_train,-1))
    y_res_train = jnp.float32(y_r_train.reshape(N * Q_train,-1))
    f_res_train = jnp.float32(f_r_train.reshape(N * Q_train,-1))
    return u_bcs_train, y_bcs_train, s_bcs_train, u_res_train, y_res_train, f_res_train 


# Generate testing data
def generate_testing_data(key, P_test, N_test, mag_scale):
    Nx = 100
    Nt = 100
    keys = jr.split(key, N_test)
    u_test, y_test, s_test = vmap(generate_one_test_data, (0, None, None))(keys, P_test, mag_scale)

    u_test = jnp.float32(u_test.reshape(N_test * P_test**2,-1))
    y_test = jnp.float32(y_test.reshape(N_test * P_test**2,-1))
    s_test = jnp.float32(s_test.reshape(N_test * P_test**2,-1))

    return u_test, y_test, s_test



class DataGenerator(data.Dataset):
    def __init__(self, u, y, s,
                 batch_size, rng_key=jr.PRNGKey(1234)):
        'Initialization'
        self.u = u # input sample
        self.y = y # location
        self.s = s # labeled data evaluated at y (solution measurements, BC/IC conditions, etc.)

        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = jr.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = jr.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx,:]
        y = self.y[idx,:]
        u = self.u[idx,:]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs
