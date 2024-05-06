import jax.random as jr
from jax.config import config
import numpy as np
import equinox as eqx
import datasets
import operators
from losses import test_loss
import matplotlib.pyplot as plt
import os


EPOCHS = 100000


XLA_PYTHON_CLIENT_PREALLOCATE=False

mag_scale = 100

# Resolution of the solution
Nx = 100
Nt = 100

N_train = 2000 # number of input samples
m = Nx   # number of input sensors
P_train = 300 # number of output sensors, 100 for each side
Q_train = 100  # number of collocation points for each input sample


config.update("jax_enable_x64", True)

key = jr.PRNGKey(1510)

# Generate training data
u_bcs_train, y_bcs_train, s_bcs_train, u_res_train, y_res_train, f_res_train = datasets.generate_training_data(key, P_train, Q_train, N_train)

# Create datasets
batch_size = 128
bcs_dataset = datasets.DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
res_dataset = datasets.DataGenerator(u_res_train, y_res_train, f_res_train, batch_size)


def run_model(NLAYERS, NUM_PROJECTION, EPOCHS):

    # Declare model
    model = operators.BHTDeepONet(branch_in_features=m, trunk_in_features=2, out_features=128, num_projection=NUM_PROJECTION, nlayers=NLAYERS)

    # branch_layers = [m, NUM_PROJECTION, NUM_PROJECTION, NUM_PROJECTION, NUM_PROJECTION, NUM_PROJECTION]
    # trunk_layers =  [2, NUM_PROJECTION, NUM_PROJECTION, NUM_PROJECTION, NUM_PROJECTION, NUM_PROJECTION]
    # model = operators.vanillaDeepONet(branch_layers, trunk_layers)


    # Enter the model name for saving (L: number of layers, NP: high dimensional features to project to)
    MODEL_NAME = f"source+tumor_L{NLAYERS}_NP{NUM_PROJECTION}"

    parent_dir = "enter directory here"
    path = os.path.join(parent_dir, MODEL_NAME)
    os.makedirs(path, exist_ok=True)



    # Start training and plot training losses
    model.train(bcs_dataset, res_dataset, nIter=EPOCHS)
    plt.figure()
    plt.plot(model.loss_log[10:]) 
    plt.show(block=False)
    plt.savefig(os.path.join(path, 'training_loss.png'))
    plt.pause(10)
    plt.close()




    # Save the model
    eqx.tree_serialise_leaves(path_or_file=os.path.join(path, 'model.eqx'),pytree=model.opt_state)
    np.save(os.path.join(path, 'loss_res.npy'), model.loss_log)
    # loss_vals = np.array(model.loss_log)
    # np.savetxt('training_loss.txt', loss_vals, fmt='%d')




    # Testing losses 
    key = jr.PRNGKey(255)
    P_test = 100
    N_test = 100
    testing_loss = test_loss(model, m, N_test, P_test, key, path, mag_scale)



    # Testing losses (side-by-side comparison with ground truth data for N_test = 1)
    key = jr.PRNGKey(275)
    P_test = 100
    N_test = 1
    testing_loss = test_loss(model, m, N_test, P_test, key, path, mag_scale)


# List of NLAYERS and NUM_PROJECTION values to experiment with
nlayer_values = [5,7]
num_projection_values = [128,256]

for NLAYERS in nlayer_values:
    for NUM_PROJECTION in num_projection_values:
        run_model(NLAYERS, NUM_PROJECTION, EPOCHS)