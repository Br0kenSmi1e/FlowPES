import jax
import jax.numpy as jnp
import pandas as pd

def make_error_loss(flow_forward, basis, data_file):
    data = pd.read_csv(data_file)
    inputs = jnp.array([eval(item) for item in data["inputs"]])
    energy = jnp.array(data["energy"])

    def get_quadratic_error(inputs, energy, basis):
        """
        basis: tau -> [phi_0, ..., phi_n]
        """
        alpha = jax.vmap(basis)(inputs)
        N = len(inputs)
        A = jnp.dot(jnp.transpose(alpha), alpha) / N
        b = jnp.dot(energy, alpha) / N
        c = jnp.mean(energy ** 2)
        return A, b, c

    def loss(params, u):
        outputs = flow_forward(params, inputs)
        A, b, c = get_quadratic_error(outputs, energy, basis)
        return jnp.dot(u, jnp.dot(A, u)) - 2 * jnp.dot(b, u) + c
    
    def reduced_loss(params):
        outputs = flow_forward(params, inputs)
        A, b, c = get_quadratic_error(outputs, energy, basis)
        u = jnp.linalg.solve(A, b)
        return c - jnp.dot(b, u)
    
    def coefficients(params):
        outputs = flow_forward(params, inputs)
        A, b, c = get_quadratic_error(outputs, energy, basis)
        u = jnp.linalg.solve(A, b)
        return u
    
    return loss, reduced_loss, coefficients