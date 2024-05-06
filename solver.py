import jax
from sourcefn import source_gaussian, source_bessel, RBF
import jax.numpy as jnp
import numpy as np
import jax.random as jr





# FDM for 1D BHT equation using Crank Nicolson discretization scheme and TDMA solver

def solve_BHT(key, Nx, Nt, P, mag_scale, length_scale, on_off):
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    k = 0.527
    v = 0.1
    rho = 1.04e-6
    cp = 3.65e6
    rho_b = 1.06e-6
    cb = 3.6e6
    wb = 8.5e-3
    T_a = 36.7 
    qm = 9.7e-3
    T_init = 37
    T_bc_left = 25
    T_bc_right = 37
    g = lambda u: 0.01*u ** 2
    dg = lambda u: 0.02 * u
    u0 = lambda x: jnp.ones_like(x)*T_init

    # Generate subkeys
    subkeys = jr.split(key, 3)

    # Generate a GP sample
    N = 512
    gp_params = (100000, length_scale)
    jitter = 1e-5
    X = jnp.linspace(xmin, xmax, N)[:,None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(N))
    gp_sample = jnp.dot(L, jr.normal(subkeys[0], (N,)))
    # Create a callable interpolation function
    f_fn = lambda x: jnp.interp(x, X.flatten(), gp_sample)
    
    

    # Create grid
    x = jnp.linspace(xmin, xmax, Nx)
    t = jnp.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    # Compute the source term

    # f = f_fn(x)
    num_curves = np.random.randint(1, 5) # 1 to 4 curves
    f = source_gaussian(num_curves, subkeys[1], mag_scale, x) + on_off*abs(f_fn(x))
    # print(f.shape)
    # f = f.at[0,0].set(0)

    # IC
    u = jnp.ones((Nx, Nt))*T_init
    u = u.at[:,0].set(u0(x))

    # Set constants to solve Au=d equation
    a = 1 + (k*dt)/(rho*cp*h2)
    b = (-1*k*dt)/(2*rho*cp*h2)
    c = b
    bp = (rho_b*cb*wb*dt)/(rho*cp)
    z = (rho_b*cb*wb*T_a + qm + f)*(dt/(rho*cp))
    # RHS
    d = jnp.zeros((Nx))
    # LHS 'A' matrix
    lower = jnp.zeros_like(d)
    lower = lower.at[1:-1].set(c) #dl[0] = 0 and BC
    lower = lower.T
    upper = jnp.zeros_like(d)
    upper = upper.at[1:-1].set(b)
    upper = upper.T
    main = jnp.ones_like(d)
    main = main.at[1:-1].set(a) #du[m - 1] = 0 and BC
    main = main.T

    for i in range(1, Nt):
        d = d.at[1:-1].set(-1*b*u[2:Nx,i-1] + (1 + b + c - bp)*u[1:Nx-1,i-1] - c*u[0:Nx-2,i-1]) + z
        d = d.at[0].set(T_bc_left)
        d = d.at[-1].set(T_bc_right)

        u_updated = jax.lax.linalg.tridiagonal_solve(lower, main, upper, jnp.expand_dims(d,1)) 
        u = u.at[:,i].set(u_updated.squeeze())

    
    UU = u

    u = f

    idx = jr.randint(subkeys[2], (P,2), 0, max(Nx,Nt))
    y = jnp.concatenate([x[idx[:,0]][:,None], t[idx[:,1]][:,None]], axis = 1)
    s = UU[idx[:,0], idx[:,1]]


    return (x, t, UU), (u, y, s)