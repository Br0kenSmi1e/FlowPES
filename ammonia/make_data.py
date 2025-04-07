import jax.numpy as jnp
from pes import ammonia_pes_

def make_data():
    with open("nh3_inv_pes.txt", "w") as f:
        f.write(",theta,phi,psi,energy\n")
    r = 1.0102908
    theta_list = jnp.arange(30, 180, 6) * jnp.pi / 180
    phi_list = jnp.arange(0, 360, 6) * jnp.pi / 180
    psi_list = jnp.arange(30, 90, 6) * jnp.pi / 180
    ndata = 0
    for theta in theta_list:
        for psi in psi_list:
            for phi in phi_list:
                energy = ammonia_pes_(r, r, r, jnp.acos(jnp.cos(psi)*jnp.cos(theta-phi)), jnp.acos(jnp.cos(psi)*jnp.cos(phi)), theta)
                with open("nh3_inv_pes.txt", "a") as f:
                    f.write(f"{ndata},{theta},{phi},{psi},{energy}\n")
                ndata += 1
            print(ndata)

make_data()
