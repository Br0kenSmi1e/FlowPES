# detailed info about NH3 PES are reported in http://dx.doi.org/10.1016/j.jms.2016.08.003 and https://doi.org/10.1063/1.1521762
# parameters taken from https://doi.org/10.1063/1.1521762, table iv, 6D-CBS*

import jax.numpy as jnp

def ammonia_pes_(r1, r2, r3, a1, a2, a3):
    """
    Ammonia PES is a 6D surface.
    Args:
        ri :: bond length of N-H_i, i = 1, 2, 3
        ai :: bond angle between N-H_{i+1} and N-H_{i+2}, the addition is under mod 3.
    """
    re = 1.0102908
    rhoe = 112.069 * jnp.pi / 180
    sinrhoe = jnp.sin(rhoe)
    a = 2.15

    xi1 = 1 - jnp.exp(-a * (r1 - re))
    xi2 = 1 - jnp.exp(-a * (r2 - re))
    xi3 = 1 - jnp.exp(-a * (r3 - re))
    xi4a = (2*a1 - a2 - a3) / jnp.sqrt(6)
    xi4b = (a2 - a3) / jnp.sqrt(6)
    sinrho = 2 * jnp.sin((a1 + a2 + a3) / 6) / jnp.sqrt(3)
    srd = sinrhoe - sinrho # sin rho deviation

    ve = -12419378.72

    v0 = 326679 * srd ** 2 - 427230 * srd ** 3 + 862091 * srd ** 4
    
    f1 = - 33571 * srd + 47047 * srd ** 2 - 346854 * srd ** 3 + 567383 * srd ** 4
    v1 = f1 * xi1

    f11 = 38755.2 - 18769 * srd + 55717 * srd ** 2 - 237488 * srd ** 3
    f13 = -422.6 + 4820 * srd + 40429 * srd ** 2 - 77362 * srd ** 3
    f4a4a = 16780.5 + 67180 * srd - 86426 * srd ** 2 + 88781 * srd ** 3
    f14a = jnp.sqrt(2/3) * (-4405.3 - 20530 * srd - 82266 * srd ** 2 + 274759 * srd ** 3)
    v2 = f11 * xi1**2 + f13 * xi1*xi3 + f4a4a * xi4a**2 + f14a * xi1*xi4a
    
    f111 = 509.9 - 10539 * srd + 28679 * srd ** 2
    f113 = -307.5 + 1655 * srd + 13188 * srd ** 2
    f123 = -236.2 + 4632 * srd
    f4a4a4a = -529.6 + 18262 * srd + 1323 * srd ** 2
    f114a = jnp.sqrt(2/3) * (-2767 - 14234 * srd)
    f134a = jnp.sqrt(1/2) * (-4994 - 17526 * srd + 36781 * srd ** 2)
    f14a4a = -2356.6 - 9494 * srd - 14168 * srd ** 2
    f24a4b = -jnp.sqrt(2) * (897 + 2698 * srd)
    v3 = f111 * xi1 ** 3 + f113 * xi1 ** 2 * xi3 + f123 * xi1 * xi2 * xi3 + f4a4a4a * xi4a ** 3 + f114a * xi1 ** 2 * xi4a + f134a * xi1 * xi3 * xi4a + f14a4a * xi1 * xi4a ** 2 + f24a4b * xi2 * xi4a * xi4b

    f1111 = 3546 - 4297 * srd
    f1113 = -487
    f1133 = -109 + 3808 * srd
    f1123 = -144 + 969 * srd
    f114a4a = -2049 + 1736 * srd
    f224a4b = -jnp.sqrt(2) * (1642 - 10368 * srd)
    f134a4a = 968 - 10275 * srd
    f134a4b = -jnp.sqrt(2) * (1320 + 8550 * srd)
    f1114a = jnp.sqrt(2/3) * (-1357 - 2978 * srd)
    f4a4a4a4a = 723 + 8850 * srd
    f1134a = jnp.sqrt(2/3) * (2528 + 1778 * srd)
    f1124b = jnp.sqrt(1/2) * (2294 - 3911 * srd)
    f14a4a4a = 260.6 + 4285 * srd
    f24a4a4b = jnp.sqrt(1/2) * (-1521 - 13132 * srd)
    v4 = f1111 * xi1 ** 4 + f1113 * xi1 ** 3 * xi3 + f1133 * xi1 ** 2 * xi3 ** 2 + f1123 * xi1 ** 2 * xi2 * xi3 + f114a4a * xi1 ** 2 * xi4a ** 2 + f224a4b * xi2 ** 2 * xi4a * xi4b + f134a4a * xi1 * xi3 * xi4a ** 2 + f134a4b * xi1 * xi3 * xi4a * xi4b + f1114a * xi1 ** 3 * xi4a + f4a4a4a4a * xi4a ** 4 + f1134a * xi1 ** 2 * xi3 * xi4a + f1124b * xi1 ** 2 * xi2 * xi4b + f14a4a4a * xi1 * xi4a ** 3 + f24a4a4b * xi2 * xi4a ** 2 * xi4b

    return v0 + v1 + v2 + v3 + v4

def ammonia_pes(inputs):
    return ammonia_pes_(*inputs)