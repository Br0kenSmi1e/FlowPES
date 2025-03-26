import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
from jax import random, vmap
from jax.example_libraries.stax import Dense, Relu, serial

import pandas as pd
import optax

def layer(transform):

    def init_fun(rng, input_dim):
        cutoff = input_dim // 2
        perm = jnp.arange(input_dim)[::-1]
        params, trans_fun = transform(rng, cutoff, 2 * (input_dim - cutoff))

        def direct_fun(params, inputs):
            lower, upper = inputs[:, :cutoff], inputs[:, cutoff:]

            # log_weight, bias = .split(2, axis=1)
            log_weight, bias = jnp.array_split(trans_fun(params, lower), 2, axis=1)
            upper = upper * jnp.exp(log_weight) + bias

            outputs = jnp.concatenate([lower, upper], axis=1)
            log_det_jacobian = log_weight.sum(-1)
            return outputs[:,perm], log_det_jacobian

        def inverse_fun(params, inputs):
            inputs = inputs[:, perm]
            lower, upper = inputs[:, :cutoff], inputs[:, cutoff:]

            log_weight, bias = jnp.array_split(trans_fun(params, lower), 2, axis=1)
            upper = (upper - bias) * jnp.exp(-log_weight)

            outputs = jnp.concatenate([lower, upper], axis=1)
            log_det_jacobian = log_weight.sum(-1)
            return outputs, log_det_jacobian

        return params, direct_fun, inverse_fun

    return init_fun

def RealNVP(transform, n: int):

    def init_fun(rng, input_dim):

        all_params, direct_funs, inverse_funs = [], [], []
        for _ in range(n):
            rng, layer_rng = random.split(rng)
            init_fun = layer(transform)
            param, direct_fun, inverse_fun = init_fun(layer_rng, input_dim)

            all_params.append(param)
            direct_funs.append(direct_fun)
            inverse_funs.append(inverse_fun)

        def feed_forward(params, apply_funs, inputs):
            log_det_jacobians = jnp.zeros(inputs.shape[:1])
            for apply_fun, param in zip(apply_funs, params):
                inputs, log_det_jacobian = apply_fun(param, inputs)
                log_det_jacobians += log_det_jacobian
            return inputs, log_det_jacobians

        def direct_fun(params, inputs):
            return feed_forward(params, direct_funs, inputs)

        def inverse_fun(params, inputs):
            return feed_forward(reversed(params), reversed(inverse_funs), inputs)

        return all_params, direct_fun, inverse_fun

    return init_fun

def harmonic_potential(tau, u0):
    return jnp.linalg.norm(tau) + u0

def make_error_loss(flow_forward, data_file):
    data = pd.read_csv(data_file, sep='\s+')
    inputs = jnp.array([[jnp.exp(-r1), jnp.exp(-r2), jnp.cos(theta*jnp.pi/180)] for r1, r2, theta in zip(data["r1"], data["r2"], data["theta"])])
    energy = jnp.array(data["energy"]) + 76
    batch_decoupled_energy = vmap(harmonic_potential, (0, None), 0)

    def loss(params, u0):
        outputs, _ = flow_forward(params, inputs)
        decoupled_energy = batch_decoupled_energy(outputs, u0)
        return jnp.mean( (decoupled_energy - energy) ** 2 )
    
    return loss


batchsize = 8192
n = 1
dim = 3
nlayers = 3
rng = random.PRNGKey(42)

def transform(rng, cutoff: int, other: int):
            net_init, net_apply = serial(Dense(16), Relu, Dense(16), Relu, Dense(other))
            in_shape = (-1, cutoff)
            out_shape, net_params = net_init(rng, in_shape)
            return net_params, net_apply

flow_init = RealNVP(transform, nlayers)

init_rng, rng = random.split(rng)
params, flow_forward, flow_inverse = flow_init(init_rng, 3)

loss = make_error_loss(flow_forward, "/Users/longli/pycode/ml4p/projects/h2opes/h2opes.txt")
value_and_grad = jax.value_and_grad(loss, argnums=(0, 1), has_aux=False)

params_optimizer = optax.adam(0.01)
params_opt_state = params_optimizer.init(params)

u0 = 0.0
u0_optimizer = optax.adam(0.01)
u0_opt_state = u0_optimizer.init(u0)

@jax.jit
def step(params, u0, params_opt_state, u0_opt_state):
    value, grad = value_and_grad(params, u0)
    params_grad, u0_grad = grad
    # u0_value, u0_grad = u0_value_and_grad(params, u0, z)
    params_updates, params_opt_state = params_optimizer.update(params_grad, params_opt_state)
    u0_updates, u0_opt_state = u0_optimizer.update(u0_grad, u0_opt_state)
    params = optax.apply_updates(params, params_updates)
    u0 = optax.apply_updates(u0, u0_updates)
    return value, params, u0, params_opt_state, u0_opt_state

loss_history = []
for i in range(5000):
    value, params, u0, params_opt_state, u0_opt_state = step(params, u0, params_opt_state, u0_opt_state)
    loss_history.append(value)
    print(i, value)
print(u0)
print(flow_inverse(params, jnp.array([[0, 0, 0]])))
