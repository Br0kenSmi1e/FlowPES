import jax
import jax.numpy as jnp
import pandas as pd

# def get_coefficients_and_error(inputs, energy, basis):
#     """
#     basis: [base], base: tau -> phi_n
#     """
#     nbasis = len(basis)
#     basis_ = [jax.vmap(base) for base in basis]
#     alpha = [basis_[n](inputs) for n in nbasis]
#     A = jnp.array([[jnp.mean(alpha[m] * alpha[n]) for n in range(m, nbasis)] for m in range(nbasis)])
#     b = jnp.array([jnp.mean(alpha[n] * energy) for n in range(nbasis)])
#     c = jnp.mean(energy ** 2)
#     u_solution = jnp.linalg.solve(A, b)
#     return u_solution, c - jnp.dot(b, u_solution)

def make_error_loss(flow_forward, basis, data_file):
    data = pd.read_csv(data_file)
    inputs = jnp.array([eval(item) for item in data["inputs"]])
    energy = jnp.array(data["energy"])

    def get_quadratic_error(inputs, energy, basis):
        """
        basis: tau -> [phi_0, ..., phi_n]
        """
        # nbasis = len(basis)
        # basis_ = [jax.vmap(base, 0, 0) for base in basis]
        # alpha = [basis_[n](inputs) for n in range(nbasis)]
        alpha = jax.vmap(basis)(inputs)
        N = len(inputs)
        # print(jnp.shape(alpha[0]), jnp.shape(alpha[1]))
        # assert jnp.shape(alpha[0])==jnp.shape(alpha[1]), f"got {jnp.shape(alpha[0])} and {jnp.shape(alpha[1])}"
        # A = jnp.array([[jnp.mean(alpha[m] * alpha[n]) if n<=m else 0 for n in range(nbasis)] for m in range(nbasis)])
        A = jnp.dot(jnp.transpose(alpha), alpha) / N
        # A = jax.array([[a[min(m, n)][max(m, n)] for n in range(nbasis)] for m in range(nbasis)])
        # b = jnp.array([jnp.mean(alpha[n] * energy) for n in range(nbasis)])
        b = jnp.dot(energy, alpha) / N
        c = jnp.mean(energy ** 2)
        # u_solution = jnp.linalg.solve(A, b)
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