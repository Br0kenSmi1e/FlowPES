import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.stats import norm

from lj_pot import lj_energy_func

def make_reinforce_loss(flow_inverse, n, dim, beta):

    batch_energy = jax.vmap(lj_energy_func)

    def loss(params, outputs):
        inputs, log_det_jacobian = flow_inverse(params, outputs)
        log_prob = norm.logpdf(inputs).sum(-1)
        entropy = log_prob - log_det_jacobian
        energy = batch_energy(outputs.reshape(-1, n, dim))
        f = entropy/beta +  energy
        f = jax.lax.stop_gradient(f)

        f_mean = jnp.mean(f)
        f_std = jnp.std(f)/jnp.sqrt(f.shape[0])

        return jnp.mean((f - f_mean) * entropy), (f_mean, f_std)
    return loss