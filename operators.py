import mlp
from jax import grad, vmap, jit
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
import itertools
import jax.numpy as jnp
import jax.random as jr
import optax
import losses
from functools import partial
from tqdm import trange, tqdm
import matplotlib.pyplot as plt



class BHTDeepONet:

    def __init__(self, branch_in_features, trunk_in_features, out_features, num_projection, nlayers):
        self.branch_params = mlp.BHTDNN.init(branch_in_features, out_features, num_projection, nlayers, key=1234)
        self.trunk_params = mlp.BHTsDNN.init(trunk_in_features, out_features, num_projection, key=4321)
        # self.trunk_params = mlp.BHTDNN.init(trunk_in_features, out_features, num_projection, nlayers, key=4321)
        params = (self.branch_params, self.trunk_params)
        
        def optimizer_fn(init_lr, decay_steps, decay_rate):
             out = optimizers.adam(optimizers.exponential_decay(init_lr, decay_steps, decay_rate))
             opt_init, opt_update, get_params = out
             return opt_init, opt_update, get_params
        
        self.opt_init, self.opt_update, self.get_params = optimizer_fn(init_lr=1e-3, decay_steps=2000, decay_rate=0.9)
        
        self.opt_state = self.opt_init(params)
        
        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # Loggers
        self.loss_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    def operator_net(self, params, source, space, time):
        branch_params = params[0]
        trunk_params = params[1]
        x = jnp.stack([space, time])
        B = mlp.BHTDNN.apply(branch_params, source)
        T = mlp.BHTsDNN.apply(trunk_params, x)
        # T = mlp.BHTDNN.apply(trunk_params, x)
        return jnp.sum(B*T)
    
    # Define ODE/PDE residual
    def residual_net(self, params, u, x, t):
        s = self.operator_net(params, u, x, t)
        s_t = grad(self.operator_net, argnums=3)(params, u, x, t)
        s_x = grad(self.operator_net, argnums=2)(params, u, x, t)
        s_xx= grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, x, t)
        
        # res = s_t - 0.05 * s_xx + 0.1 * s_x
        res = (1.04e-6 * 3.65e+6 * s_t - 0.527 * s_xx - 1.06e-6 * 3.6e+6 * 8.5e-3 * (36.7 - s) - 9.7e-3) / 100
        # print(res)
        return res

    # Define boundary loss
    def loss_bcs(self, params, batch):
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])

        # Compute loss
        loss = losses.loss_function(outputs.flatten(), s_pred) #+ self.l2_reg(s_pred)
        return loss

    # Define residual loss
    def loss_res(self, params, batch):
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.residual_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])

        # Compute loss
        loss = losses.loss_function(outputs.flatten(), pred) #+ losses.l2_reg(pred)
        return loss

    # Define total loss
    def loss(self, params, bcs_batch, res_batch):
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss = loss_bcs + loss_res
        return loss
    
    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, bcs_batch, res_batch)
        g = optimizers.clip_grads(g, 1)
        return self.opt_update(i, g, opt_state)
    
    
    # @partial(jit, static_argnums=(0,))
    # def step(self, i, opt_state, bcs_batch, res_batch):
    #     params = optax.OptState
    #     g = grad(self.loss)(params, bcs_batch, res_batch)
    #     return self.opt_update(i, g, opt_state)
    
    # @partial(jit, static_argnums=(0,))
    # def step(self, i, opt_state, bcs_batch, res_batch):
    #     loss_value, grads = jax.value_and_grad(self.loss)(self.params, bcs_batch, res_batch)
    #     updates, opt_state = self.opt.update(grads, opt_state, self.params)
    #     params = optax.apply_updates(params, updates)
    #     return params, opt_state, loss_value

    

    # Optimize parameters in a loop
    def train(self, bcs_dataset, res_dataset, nIter = 10000):
        # Define data iterators
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)


        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            bcs_batch= next(bcs_data)
            res_batch = next(res_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, bcs_batch, res_batch)

            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, bcs_batch, res_batch)
                loss_bcs_value = self.loss_bcs(params, bcs_batch)
                loss_res_value = self.loss_res(params, res_batch)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                # Print losses
                pbar.set_postfix({'Loss': loss_value,
                                  'loss_bcs' : loss_bcs_value,
                                  'loss_physics': loss_res_value})
                
        
                
     

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return r_pred
    




class vanillaDeepONet:
    def __init__(self, branch_layers, trunk_layers):
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = mlp.vanillaMLP(branch_layers, activation=jnp.tanh)
        self.trunk_init, self.trunk_apply = mlp.vanillaMLP(trunk_layers, activation=jnp.tanh)

        # Initialize
        branch_params = self.branch_init(rng_key = jr.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key = jr.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3,
                                                                      decay_steps=2000,
                                                                      decay_rate=0.9))
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # Loggers
        self.loss_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    # Define DeepONet architecture
    def operator_net(self, params, u, x, t):
        branch_params, trunk_params = params
        y = jnp.stack([x, t])
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = jnp.sum(B * T)
        return outputs

    # Define ODE/PDE residual
    def residual_net(self, params, u, x, t):
        s = self.operator_net(params, u, x, t)
        s_t = grad(self.operator_net, argnums=3)(params, u, x, t)
        s_x = grad(self.operator_net, argnums=2)(params, u, x, t)
        s_xx= grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, x, t)

        # res = s_t - 0.05 * s_xx + 0.1 * s_x
        res = (1.04e-6 * 3.65e+6 * s_t - 0.527 * s_xx - 1.06e-6 * 3.6e+6 * 8.5e-3 * (36.7 - s) - 9.7e-3) / 100
        return res

    # Define boundary loss
    def loss_bcs(self, params, batch):
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])

        # Compute loss
        loss = jnp.mean((outputs.flatten() - s_pred)**2)
        return loss

    # Define residual loss
    def loss_res(self, params, batch):
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.residual_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])

        # Compute loss
        loss = jnp.mean((outputs.flatten() - pred)**2)
        return loss

    # Define total loss
    def loss(self, params, bcs_batch, res_batch):
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss = loss_bcs + loss_res
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, bcs_dataset, res_dataset, nIter = 10000):
        # Define data iterators
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            bcs_batch= next(bcs_data)
            res_batch = next(res_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, bcs_batch, res_batch)

            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, bcs_batch, res_batch)
                loss_bcs_value = self.loss_bcs(params, bcs_batch)
                loss_res_value = self.loss_res(params, res_batch)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                # Print losses
                pbar.set_postfix({'Loss': loss_value,
                                  'loss_bcs' : loss_bcs_value,
                                  'loss_physics': loss_res_value})

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return r_pred