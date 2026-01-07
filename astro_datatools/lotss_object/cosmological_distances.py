from astropy.cosmology import Planck18 as cosmo
from astropy.constants import c
import numpy as np
from scipy.integrate import quad

# Get cosmological parameters
OMEGA_M = cosmo.Om0                     # Total matter density parameter (dark + baryonic)
OMEGA_B = cosmo.Ob0                     # Baryonic matter density parameter
OMEGA_DM = OMEGA_M - OMEGA_B            # Dark matter density
OMEGA_LAMBDA = cosmo.Ode0               # Dark energy density
OMEGA_RAD = cosmo.Ogamma0 + cosmo.Onu0  # Radiation (photons + neutrinos)
OMEGA_CURV = cosmo.Ok0                  # Curvature
H0 = cosmo.H0.value                     # Hubble constant in km/s/Mpc
c_KMS = c.to('km/s').value              # Speed of light in km/s

print(f"Cosmological Parameters used for calculating distances:")
print(f"Omega_M: {OMEGA_M}")
print(f"Omega_B: {OMEGA_B}")
print(f"Omega_DM: {OMEGA_DM}")
print(f"Omega_Lambda: {OMEGA_LAMBDA}")
print(f"Omega_Rad: {OMEGA_RAD}")
print(f"Omega_Curv: {OMEGA_CURV}")
print(f"H0: {H0} km/s/Mpc")
print(f"c (speed of light): {c_KMS} km/s")

def hubble_distance():
    """Calculate the Hubble distance D_H.
    
    The formula used is:
    D_H = c / H0

    Units:
    - c in km/s
    - H0 in km/s/Mpc
    - D_H in Mpc
    """
    return c_KMS / H0  # in Mpc

def hubble_parameter(z):
    """
    Calculate the Hubble parameter H(z) at redshift z.
    
    The formula used is:
    H(z) = H0 * sqrt(OMEGA_M * (1 + z)^3 + OMEGA_RAD * (1 + z)^4 + OMEGA_LAMBDA + OMEGA_CURV * (1 + z)^2)

    Units:
    - H0 in km/s/Mpc
    - H(z) in km/s/Mpc
    """
    return H0 * (OMEGA_M * (1 + z)**3 + OMEGA_RAD * (1 + z)**4 + OMEGA_LAMBDA + OMEGA_CURV * (1 + z)**2)**0.5

def normalized_hubble_parameter(z):
    """
    Calculate the normalized Hubble parameter.
     
    The formula used is:
    E(z) = H(z) / H0 at redshift z.

    Units:
    - H(z) in km/s/Mpc
    - H0 in km/s/Mpc
    - E(z) is dimensionless
    """
    return hubble_parameter(z) / H0

def comoving_distance(z):
    """
    Calculate the comoving distance D_C to redshift z.
    
    The formula used is:
    D_C = D_H * integral from 0 to z of dz' / E(z')

    Units:
    - D_H in Mpc
    - E(z) is dimensionless
    - D_C in Mpc
    """
    D_H = hubble_distance()
    
    integral, _ = quad(
        lambda z_prime: 1.0 / normalized_hubble_parameter(z_prime),
        a=0,
        b=z
    )
    
    return D_H * integral

def transverse_comoving_distance(z):
    """
    Calculate the transverse comoving distance D_M to redshift z.
    
    The formula used is:
    D_M = D_C for flat universe (OMEGA_CURV = 0)
    D_M = (c / (H0 * sqrt(OMEGA_CURV))) * sinh(sqrt(OMEGA_CURV) * (H0/c) * D_C) for OMEGA_CURV > 0
    D_M = (c / (H0 * sqrt(-OMEGA_CURV))) * sin(sqrt(-OMEGA_CURV) * (H0/c) * D_C) for OMEGA_CURV < 0

    Units:
    - D_C in Mpc
    - D_M in Mpc
    """
    D_C = comoving_distance(z)
    
    if OMEGA_CURV > 0:
        sqrt_OK = OMEGA_CURV**0.5
        d_H = hubble_distance()
        return (d_H / sqrt_OK) * np.sinh(sqrt_OK * D_C / d_H)
    elif OMEGA_CURV < 0:
        sqrt_OK = (-OMEGA_CURV)**0.5
        return (d_H / sqrt_OK) * np.sin(sqrt_OK * D_C / d_H)
    else:
        return D_C
    
def angular_diameter_distance(z):
    """
    Calculate the angular diameter distance D_A to redshift z.
    
    The formula used is:
    D_A = D_M / (1 + z)

    Units:
    - D_M in Mpc
    - D_A in Mpc
    """
    return cosmo.angular_diameter_distance(z).value

def luminosity_distance(z):
    """
    Calculate the luminosity distance D_L to redshift z.
    
    The formula used is:
    D_L = (1 + z) * D_M

    Units:
    - D_M in Mpc
    - D_L in Mpc
    """
    return cosmo.luminosity_distance(z).value


if __name__ == "__main__":
    z = np.linspace(0, 10, 100)
    ang_diameter_dist = np.array([angular_diameter_distance(zi) for zi in z])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(z, ang_diameter_dist, label='Angular Diameter Distance $D_A$', color='blue')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Distance (Mpc)')
    plt.title('Angular Diameter Distance vs Redshift')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()
