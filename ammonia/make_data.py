import jax.numpy as jnp
from pes import ammonia_pes_

def make_data():
    with open("nh3_inv_pes.txt", "w") as f:
        f.write(",inputs,energy\n")
    r = 1.0102908
    theta_list = jnp.arange(30, 180, 6) * jnp.pi / 180
    phi_list = jnp.arange(30, 180, 6) * jnp.pi / 180
    psi_list = jnp.arange(30, 150, 6) * jnp.pi / 180
    ndata = 0
    for theta in theta_list:
        for psi in psi_list:
            for phi in phi_list:
                a1 = jnp.acos(jnp.sin(theta)*jnp.sin(psi)*jnp.cos(phi)+jnp.cos(theta)*jnp.cos(phi))
                energy = ammonia_pes_(r, r, r, a1, psi, theta) / 1e4
                with open("nh3_inv_pes.txt", "a") as f:
                    f.write(f'{ndata},"{[theta.item(), phi.item(), psi.item()]}",{energy}\n')
                ndata += 1
            print(ndata)

def make_inv_pot_data():
    with open("nh3_inv_pes.txt", "w") as f:
        f.write(",inputs,energy\n")
    T = jnp.arange(50, 130, 1) * jnp.pi / 180
    R = jnp.arange(0.8, 1.2, 0.01)
    ndata = 0
    for t in T:
        a = 2 * jnp.asin(jnp.sqrt(3)*jnp.sin(t)/2)
        for r in R:
            energy = ammonia_pes_(r, r, r, a, a, a) / 1e4
            with open("nh3_inv_pes.txt", "a") as f:
                f.write(f'{ndata},"{[t.item(), r.item()]}",{energy}\n')
            ndata += 1
        print(ndata)

if __name__ == "__main__":
    make_inv_pot_data()
