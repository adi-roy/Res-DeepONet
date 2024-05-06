import equinox as eqx
import jax
from layer import NiNLayer as MLPLayer
import jax.random as jr
import jax.numpy as jnp

class BHTnet(eqx.Module):
    layers: list
    # weights: jax.Array
    # bias: jax.Array
    # linear1: MLPLayer
    # linear2: MLPLayer
    # linear3: MLPLayer
    # linear4: MLPLayer
    # linear5: MLPLayer
    #dropout: eqx.nn.Dropout
    stem: eqx.nn.Linear
    regression: eqx.nn.Linear

    def __init__(self, input_features, output_features, num_projection, nlayers, key):
        # key1 = jr.PRNGKey(key)
        subkeys = jr.split(key,15)
        self.layers = []
        self.stem = eqx.nn.Linear(in_features=input_features, out_features=num_projection, key = subkeys[0])
        self.regression = eqx.nn.Linear(in_features=num_projection, out_features=output_features, key = subkeys[1])

        for _ in range(nlayers):
            self.layers.append(MLPLayer(input_features, output_features, num_projection, factor = 4))
        # self.linear1 = MLPLayer(input_features, output_features, num_projection, factor = 4)
        # self.linear2 = MLPLayer(input_features, output_features, num_projection, factor = 4)
        # self.linear3 = MLPLayer(input_features, output_features, num_projection, factor = 4)
        # self.linear4 = MLPLayer(input_features, output_features, num_projection, factor = 4)
        # self.linear5 = MLPLayer(input_features, output_features, num_projection, factor = 4)
        # #self.dropout = eqx.nn.Dropout(p=0.4, inference=False)

def branchinitializer(weight: jax.Array, key) -> jax.Array:
    init = jax.nn.initializers.he_normal() # Xavier init of weights
    out = init(key, weight.shape)
    return out

def branchMLP(model, init_fn, key): # function to replace FCN random weights with Xavier init weights
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x)]
    weights = get_weights(model)
    new_weights = [init_fn(weight, subkey)
                    for weight, subkey in zip(weights, jr.split(key, len(weights)))]
    return eqx.tree_at(get_weights, model, new_weights)

class BHTDNN(BHTnet):

    def __init__(self, in_features, out_features, num_projection, nlayers, key):
        super().__init__(in_features, out_features, num_projection, nlayers, key)
    def init(in_features: int, out_features: int, num_projection: int, nlayers: int, key, init=branchinitializer):
        key1 = jr.PRNGKey(key)
        subkeys = jr.split(key1,2)
        output =  branchMLP(BHTnet(in_features, out_features, num_projection, nlayers, subkeys[0]), init, subkeys[1])
        # output =  BHTnet(in_features, out_features, num_projection, nlayers, subkeys[0])
        # self.weight = output.layers[:].weight
        # self.bias = output.layers[:].bias
        return output
    def apply(output, x):
        # x = output.linear1(x) + x
        # x = output.linear2(x) + x 
        # x = output.linear3(x) + x
        # x = output.linear4(x) + x
        # x = output.linear5(x) + x
        x = output.stem(x)
        y = x   # experimental
        for layer in output.layers:
            x = layer(x)
        x = x #+ y # skip connection between stem and regression
        x = output.regression(x) #+ y   
        return x
    


class FCN(eqx.Module):  
    # layers: list
    # weights: jax.Array
    # bias: jax.Array
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear
    linear4: eqx.nn.Linear
    linear5: eqx.nn.Linear
    stem: eqx.nn.Linear
    regression: eqx.nn.Linear

    def __init__(self, input_features, output_features, num_projection, key):
        # key1 = jr.PRNGKey(key)
        subkeys = jr.split(key,7)
        # self.layers = [eqx.nn.Linear(in_features=input_features, out_features=50, key=subkeys[0]),
        #                eqx.nn.Linear(in_features=50, out_features=50, key=subkeys[1]),
        #                eqx.nn.Linear(in_features=50, out_features=50, key=subkeys[2]),
        #                eqx.nn.Linear(in_features=50, out_features=50, key=subkeys[3]),
        #                eqx.nn.Linear(in_features=50, out_features=output_features, key=subkeys[4])]
        self.stem = eqx.nn.Linear(input_features, num_projection, key=subkeys[0])
        self.linear1 = eqx.nn.Linear(num_projection, num_projection, key=subkeys[1])
        self.linear2 = eqx.nn.Linear(num_projection, num_projection // 2, key=subkeys[2])
        self.linear3 = eqx.nn.Linear(num_projection // 2, num_projection // 4, key=subkeys[3])
        self.linear4 = eqx.nn.Linear(num_projection // 4, num_projection // 2, key=subkeys[4])
        self.linear5 = eqx.nn.Linear(num_projection // 2, num_projection, key=subkeys[5])
        self.regression = eqx.nn.Linear(num_projection, output_features, key=subkeys[6])


def trunkinitializer(weight: jax.Array, key) -> jax.Array:
    init = jax.nn.initializers.glorot_normal() # Xavier init of weights
    out = init(key, weight.shape)
    return out

def trunkMLP(model, init_fn, key): # function to replace FCN random weights with Xavier init weights
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x)]
    weights = get_weights(model)
    new_weights = [init_fn(weight, subkey)
                    for weight, subkey in zip(weights, jr.split(key, len(weights)))]
    return eqx.tree_at(get_weights, model, new_weights)


class BHTsDNN(FCN):

    def __init__(self, in_features, out_features, key):
        super().__init__(in_features, out_features, key)
    def init(in_features, out_features, num_projection, key, init=trunkinitializer):
        key1 = jr.PRNGKey(key)
        subkeys = jr.split(key1,2)
        output =  trunkMLP(FCN(in_features, out_features, num_projection, subkeys[0]), init, subkeys[1])
        # output = FCN(in_features, out_features, num_projection, subkeys[0])
        # self.weight = output.layers[:].weight
        # self.bias = output.layers[:].bias
        return output
    def apply(output, x):
        # for layer in output.layers[:-1]:
        #     x = jax.nn.tanh(layer(x))
        # out = output.layers[-1](x)
        #print(out.shape)
        x = output.stem(x)
        y = jax.nn.tanh(output.linear1(x))
        z = jax.nn.tanh(output.linear2(y))
        x = jax.nn.tanh(output.linear3(z))
        x = jax.nn.tanh(output.linear4(x)) #+ z # ablate skip connections
        x = jax.nn.tanh(output.linear5(x)) #+ y 
        # x = jnp.multiply(jax.nn.tanh(output.linear4(x)),z) + jnp.multiply((1-z),jax.nn.tanh(output.linear4(x)))
        # x = jnp.multiply(jax.nn.tanh(output.linear5(x)),y) + jnp.multiply((1-y),jax.nn.tanh(output.linear5(x)))
        x = output.regression(x) 
        return x



def vanillaMLP(layers, activation=jax.nn.relu):
  ''' Vanilla MLP'''
  def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = jr.split(key)
          glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * jr.normal(k1, (d_in, d_out))
          b = jnp.zeros(d_out)
          return W, b
      key, *keys = jr.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
  def apply(params, inputs):
      for W, b in params[:-1]:
          #print(inputs.shape)
          #print(W.shape)
          outputs = jnp.dot(inputs, W) + b
          inputs = activation(outputs)
      W, b = params[-1]
      outputs = jnp.dot(inputs, W) + b
      return outputs
  return init, apply