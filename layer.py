import equinox as eqx
import jax.random as jr
import jax


class NiNLayer(eqx.Module):
    layers: list
    # stem: eqx.nn.Linear
    # regression: eqx.nn.Linear
    # dropout: eqx.nn.Dropout

    def __init__(self, input_features, output_features, num_projection, factor):

        # self.stem = eqx.nn.Linear(in_features=input_features, out_features=num_projection, key=jr.PRNGKey(11))
        self.layers = [eqx.nn.Linear(in_features=num_projection, out_features=num_projection//factor, key=jr.PRNGKey(23)),
                       eqx.nn.Lambda(eqx.nn.PReLU()),
                    #    jax.nn.tanh(),
                       eqx.nn.GroupNorm(groups=16, channels=None, channelwise_affine=False),
                       eqx.nn.Linear(in_features=num_projection//factor, out_features=num_projection//factor, key=jr.PRNGKey(45)),
                       eqx.nn.Lambda(eqx.nn.PReLU()),
                    #    jax.nn.tanh(),
                       eqx.nn.GroupNorm(groups=16, channels=None, channelwise_affine=False),
                       eqx.nn.Linear(in_features=num_projection//factor, out_features=num_projection, key=jr.PRNGKey(78))
                       ]
        # self.regression = eqx.nn.Linear(in_features=num_projection, out_features=output_features, key=jr.PRNGKey(98))
   

    def __call__(self, x):
        dropout = eqx.nn.Dropout(p=0.1, inference=False)
        # x = self.stem(x)
        y = dropout(x, key=jr.PRNGKey(222))
        for layer in self.layers[:-1]:
            y = layer(y)
        x = self.layers[-1](y) + x 
        # return self.regression(x)
        return x