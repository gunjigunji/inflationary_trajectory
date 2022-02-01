import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from fractions import Fraction
from scipy.integrate import solve_ivp
from matplotlib.backends.backend_pdf import PdfPages


# --- variables for sympy ---
phi = sym.Symbol("phi")
s = sym.Symbol("s")
dphi = sym.Symbol("dphi")
ds = sym.Symbol("ds")
t = sym.symbols("t")  # Convert to sympy form to use "t" as the lambdify argment


# --- parameters ---
# Adopting Planck unit: Mpl = 1
# Reduced Planck mass
MPL = Fraction(2.435 * pow(10, 18))

# Model parameters consistent with the CMB observations
chi = 0
alp = Fraction(2, 3)
lam = pow(10, -3)
g = 1
q = 1
xi = Fraction((0.6 * pow(10, 16) / MPL) ** 2)
xit = xi / (3 * alp * q)


# --- functions ---
# Phi
def Phi(phi, s):
    return -3 + s**2 / 2 + (1 + chi) * phi**2 / 2


# -Phi/3
def mPhiov3(phi, s):
    return Fraction(-1, 3) * Phi(phi, s)


# Psi
def Psi(phi, s):
    return (
        (mPhiov3(phi, s)) ** (2 - 3 * alp)
        * lam**2
        / (2 * alp**2 * q * g**2 * xi)
        * phi**2
    )


# Kahler potenaial K
def K(phi, s):
    return -3 * alp * sym.log(Fraction(-1, 3) * Phi(phi, s))


# scalar potential V
def V(phi, s):
    VFE = (
        (mPhiov3(phi, s)) ** (1 - 3 * alp) * 1 / alp * lam**2 / 4 * phi**2 * s**2
    )
    VDE = g**2 / 8 * ((mPhiov3(phi, s)) ** (-1) * alp * q * s**2 - 2 * xi) ** 2
    return VFE + VDE


# dV/dphi
def dVovdphi(argphi, args):
    return sym.diff(V(phi, s), phi).subs([(phi, argphi), (s, args)])


# dV/ds
def dVovds(argphi, args):
    return sym.diff(V(phi, s), s).subs([(phi, argphi), (s, args)])


# d^2K/dphi^2
def Kpp(argphi, args):
    return sym.diff(K(phi, s), phi, 2).subs([(phi, argphi), (s, args)])


# d^2K/ds^2
def Kss(argphi, args):
    return sym.diff(K(phi, s), s, 2).subs([(phi, argphi), (s, args)])


# d(d^2K/dphi^2)/dphi
def Kppp(argphi, args):
    return sym.diff(K(phi, s), phi, 3).subs([(phi, argphi), (s, args)])


# d(d^2K/ds^2)/ds
def Ksss(argphi, args):
    return sym.diff(K(phi, s), s, 3).subs([(phi, argphi), (s, args)])


# d(d^2K/dphi^2)/ds
def Kpps(argphi, args):
    return sym.diff(sym.diff(K(phi, s), phi, 2), s).subs([(phi, argphi), (s, args)])


# d(d^2K/ds^2)/dphi
def Kssp(argphi, args):
    return sym.diff(sym.diff(K(phi, s), s, 2), phi).subs([(phi, argphi), (s, args)])


# apploximate local minimum s_min^2
def s2appr(phi):
    return (
        2
        * 3
        * xit
        * (mPhiov3(phi, 0))
        * (1 - Psi(phi, 0))
        / (1 + xit * (1 - Psi(phi, 0)))
    )


# energy density
def rho(phi, s, dphi, ds):
    return V(phi, s) + Kpp(phi, s) * dphi**2 / 2 + Kss(phi, s) * ds**2 / 2


# Hubble parameter H
def H(phi, s, dphi, ds):
    return sym.sqrt(rho(phi, s, dphi, ds) * Fraction(1, 3))


# --- initial conditions ---
# Critical-point value
phic0sq = 2 * alp**2 * q * g**2 * xi / (lam**2)

# Deviation
delta = 1 * pow(10, -1)

# Initial value of inflaton field
phi_init = sym.sqrt(phic0sq * (1 - delta))

# Initial value of waterfall field
s_init = sym.sqrt(s2appr(phi_init))

# Initial value of dphi/dt
dphi_init = 0

# Initial value of ds/dt
ds_init = 0


# --- main ---
if __name__ == "__main__":
    dphiovdt = sym.lambdify(
        (t, phi, s, dphi, ds),
        -3 * H(phi, s, dphi, ds) * dphi
        - 1
        / Kpp(phi, s)
        * (
            dVovdphi(phi, s)
            + Kppp(phi, s) * dphi**2
            + Kpps(phi, s) * dphi * ds
            - 1 / 2 * (Kppp(phi, s) * dphi**2 + Kssp(phi, s) * ds**2)
        ),
        "scipy",
    )
    dsovdt = sym.lambdify(
        (t, phi, s, dphi, ds),
        -3 * H(phi, s, dphi, ds) * ds
        - 1
        / Kss(phi, s)
        * (
            dVovds(phi, s)
            + Ksss(phi, s) * ds**2
            + Kssp(phi, s) * ds * dphi
            - 1 / 2 * (Ksss(phi, s) * ds**2 + Kpps(phi, s) * dphi**2)
        ),
        "scipy",
    )
    ddphiovdt2 = sym.lambdify((t, phi, s, dphi, ds), dphi, "scipy")
    ddsovdt2 = sym.lambdify((t, phi, s, dphi, ds), ds, "scipy")

    def rhss(t, fields):
        phi, s, dphi, ds = fields
        dphidt = dphiovdt(t, phi, s, dphi, ds)
        dsdt = dsovdt(t, phi, s, dphi, ds)
        ddphidt2 = ddphiovdt2(t, phi, s, dphi, ds)
        ddsdt2 = ddsovdt2(t, phi, s, dphi, ds)
        return [dphidt, dsdt, ddphidt2, ddsdt2]

    # --- initial conditins ---
    init = [phi_init, s_init, dphi_init, ds_init]

    t_min = 0
    t_max = 1.5 * 10**12
    t_span = [t_min, t_max]
    n_t = 10**4
    t_eval = np.linspace(t_min, t_max, n_t)

    sol = solve_ivp(rhss, t_span, init, method="Radau", dense_output=True)
    sol_sm = sol.sol(t_eval)

    # --- plot ---
    N = 100
    fig, ax = plt.subplots()
    phi_list = np.linspace(0, float(phi_init), N)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$ s$")
    ax.set_title("Trajectory")
    ax.grid(color="gray", linestyle=":")
    ax.plot(sol_sm[0, :], sol_sm[1, :], color="tab:blue", label="num")
    # ax.plot(sol.y[0,:], sol.y[1,:], color="tab:blue", linestyle='-.', label="num")
    ax.plot(
        phi_list,
        [sym.sqrt(s2appr(phi_list[k])) for k in range(N)],
        color="tab:orange",
        linestyle=":",
        label="app",
    )
    ax.legend(loc=0)
    fig.tight_layout()

    pdf = PdfPages("figures/trajectory.pdf")
    pdf.savefig(fig)
    pdf.close()

    plt.show()
