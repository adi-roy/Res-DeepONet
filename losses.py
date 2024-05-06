import optax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import datasets
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os


# MSE loss
def loss_function(val_true, val_predicted):
    loss = optax.squared_error(val_predicted, val_true).mean()
    return loss

# l2 regularization    
def l2_reg(params, reg_lambda=0.2):
    l2_norm = jnp.linalg.norm(params, ord=2)
    return reg_lambda * l2_norm

# Testing loss
def test_loss(model, m, N_test, P_test, key, path, mag_scale):
    Nx = m
    Nt = m
    u_test, y_test, s_test = datasets.generate_testing_data(key, P_test, N_test, mag_scale)
    inference_model = eqx.nn.inference_mode(model)
    params = model.get_params(model.opt_state)
    x = jnp.linspace(0, 1, Nx)
    t = jnp.linspace(0, 1, Nt)
    XX, TT = jnp.meshgrid(x, t)
    error_test = []
    for k in range(N_test):
        s_test_ = s_test[k*P_test**2:(k+1)*P_test**2,:]
        u_test_ = u_test[k*P_test**2:(k+1)*P_test**2,:]
        y_test_ = y_test[k*P_test**2:(k+1)*P_test**2,:]
        S_test = griddata(y_test_, s_test_.flatten(), (XX,TT), method='cubic')

        s_pred = inference_model.predict_s(params, u_test_, y_test_)
        S_pred = griddata(y_test_, s_pred.flatten(), (XX,TT), method='cubic')
        error_s = jnp.linalg.norm(S_test - S_pred) / jnp.linalg.norm(S_test) 
        error_test.append(error_s)
    testing_loss = jnp.stack(error_test)

    if N_test > 1:
        plt.figure(figsize=(10,5))
        plt.plot(testing_loss)
        plt.show(block=False)
        plt.savefig(os.path.join(path, 'testing_loss.png'))
        plt.pause(20)
        plt.close()
        test_loss_mean = jnp.mean(testing_loss)
        print('Testing loss for ' + str(N_test) + ' samples = '+ str(test_loss_mean))
    else:
        test_loss_mean = testing_loss
        print('Testing loss for 1 sample = ' + str(test_loss_mean))
        
        # Plot
        fig = plt.figure(figsize=(18,5))
        plt.subplot(1,3,1)
        plt.pcolor(XX,TT, S_test, cmap='seismic')
        plt.colorbar()
        plt.xlabel('')
        plt.ylabel('')
        plt.title('Exact ')
        plt.tight_layout()

        plt.subplot(1,3,2)
        plt.pcolor(XX,TT, S_pred, cmap='seismic')
        plt.colorbar()
        plt.xlabel('')
        plt.ylabel('')
        plt.title('Predict ')
        plt.tight_layout()

        plt.subplot(1,3,3)
        plt.pcolor(XX,TT, S_pred - S_test, cmap='seismic')
        plt.colorbar()
        plt.xlabel('')
        plt.ylabel('')
        plt.title('Absolute error')
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(os.path.join(path, 'comparison.png'))
        plt.pause(20)
        plt.close()

    return test_loss_mean


