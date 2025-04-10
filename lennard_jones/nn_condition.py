# inputs[:,-1]:: beta, as a parameter
import jax
import jax.numpy as jnp
from jax.example_libraries.stax import Dense, serial

def layer(transform):

    def init_fun(rng, input_dim):
        input_var_dim = input_dim - 1
        cutoff = input_var_dim // 2
        perm = jnp.arange(input_var_dim)[::-1]
        params, trans_fun = transform(rng, cutoff + 1, 2 * (input_var_dim - cutoff))

        def direct_fun(params, inputs):
            lower, upper, conParam = inputs[:, 0:cutoff], inputs[:, cutoff:input_var_dim], inputs[:, -1].reshape(-1, 1)

            log_weight, bias = jnp.array_split(trans_fun(params, jnp.concatenate([lower, conParam], axis=1)), 2, axis=1)
            upper = upper * jnp.exp(log_weight) + bias

            outputs = jnp.concatenate([lower, upper], axis=1)
            outputs = jnp.concatenate([outputs[:, perm], conParam], axis=1)
            log_det_jacobian = log_weight.sum(-1)
            return outputs, log_det_jacobian

        def inverse_fun(params, inputs):
            conParam = inputs[:, -1].reshape(-1, 1)
            inputs = inputs[:, perm]
            lower, upper = inputs[:, :cutoff], inputs[:, cutoff:]

            log_weight, bias = jnp.array_split(trans_fun(params, jnp.concatenate([lower, conParam], axis=1)), 2, axis=1)
            upper = (upper - bias) * jnp.exp(-log_weight)

            outputs = jnp.concatenate([lower, upper, conParam], axis=1)
            log_det_jacobian = log_weight.sum(-1)
            return outputs, log_det_jacobian

        return params, direct_fun, inverse_fun

    return init_fun

def RealNVP(transform, n: int):

    def init_fun(rng, input_dim):

        all_params, direct_funs, inverse_funs = [], [], []
        for _ in range(n):
            rng, layer_rng = jax.random.split(rng)
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

def make_transform(hidden_dim: int, activation):
    def transform(rng, cutoff: int, other: int):
        net_init, net_apply = serial(Dense(hidden_dim), activation, Dense(hidden_dim), activation, Dense(other))
        in_shape = (-1, cutoff)
        out_shape, net_params = net_init(rng, in_shape)
        return net_params, net_apply
    return transform