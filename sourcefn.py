import jax.numpy as jnp
from jax import random as jr
from jax.scipy.special import bessel_jn
from jax import vmap
from jax import lax



# Source term generator for BHT equation #

def gaussian_generator(key, mag_scale, x):

    # key = jr.PRNGKey(2)
    subkeys = jr.split(key, 3)
    zeta = jr.uniform(subkeys[0], minval=1, maxval=mag_scale)
    stdev = jr.uniform(subkeys[1], minval=0.1, maxval=0.2)
    mean = jr.uniform(subkeys[2], minval=0.1, maxval=0.9)
    # Define the range of x values
    # x = jnp.linspace(0, 1, 100)

    # Calculate the corresponding probability density values (bell curve)
    output_dist = jnp.zeros_like(x)
    output_dist = zeta*(1.0 / (stdev *jnp.sqrt(2 *jnp.pi))) *jnp.exp(-0.5 * ((x - mean) / stdev) ** 2)
    # output_dist =jnp.clip(output_dist, 0, mag_scale*2)

    return output_dist

def source_gaussian(num_curves, key, mag_scale, x):
    subkeys = jr.split(key, num_curves)
    y = jnp.zeros_like(x)
    for i in range(num_curves):
        y += gaussian_generator(subkeys[i], mag_scale, x)
    return y


# GRF sampling #
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - \
            jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs**2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)


def bessel_generator(key, mag_scale, r):

    # Set Bessel beam parameters
    subkeys = jr.split(key, 3)
    w0 = jr.uniform(subkeys[2], minval=0.1, maxval=0.5)  # Waist of the Bessel beam
    # w0 = 0.2
    m = 3    # Order of the Bessel function
    arg = m * r / w0
    # intensity = jnp.zeros_like(r)
    # intensity = lax.cond(arg == 0, lambda x: 1.0, lambda x: mag_scale * (2 * bessel_jn(x, v=m) / x)**2, operand=arg)
    w = w0 * jnp.sqrt(1 + (m * r / w0)**2)  # Bessel beam width
    intensity = jnp.zeros_like(r)
    intensity = mag_scale*(2 * bessel_jn(m * r / w0, v=m) / (m * r / w0))**2  # Bessel beam intensity
    return (intensity * jnp.exp(-(r / w)**2)).T[:,-1]  # Gaussian envelope



def source_bessel(num_curves, key, mag_scale, x):
    subkeys = jr.split(key, num_curves)
    y = jnp.zeros_like(x)
    # y += vmap(lambda ri: bessel_generator(subkeys[1], mag_scale, ri)(x))
    for i in range(num_curves):
        # y += vmap(lambda ri: bessel_generator(subkeys[i], mag_scale, ri)(x))
        y += bessel_generator(subkeys[i], 100*mag_scale, x)
    y = y.at[0].set(0)
    return y