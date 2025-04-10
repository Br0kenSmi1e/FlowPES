import jax.numpy as jnp

def lj_energy_func(x):
    """
    x = [x_1, x_2, ..., x_n] is a list of coordinates of particles.
    each x_k = [x_k1, ..., x_kd] is a list of Cartesian coordinates.
    """
    i, j = jnp.triu_indices(len(x), k=1)
    r = jnp.linalg.norm(x[i] - x[j])
    v_e = jnp.sum(jnp.linalg.norm(x))
    return jnp.sum(r**-12 - r**-6) + 0.01 * v_e

if __name__ == "__main__":
    import jax
    import matplotlib.pyplot as plt
    distance = jnp.linspace(1, 2, 100)
    inputs = jnp.array([[[-d/2, 0, 0], [d/2, 0, 0]] for d in distance])
    energy = jax.vmap(lj_energy_func)(inputs)
    plt.plot(distance, energy)
    plt.show()
