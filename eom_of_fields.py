import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from fractions import Fraction
from scipy.integrate import solve_ivp
from matplotlib.backends.backend_pdf import PdfPages

# Adopting Planck unit: Mpl = 1

# --- variables for sympy ---
phi = sym.Symbol('phi')
s = sym.Symbol('s')
t = sym.symbols('t') # convert to sympy form to use "t" as the lambdify argment

# --- global parameters ---
chi = 0
alp = Fraction(2,3)
MPL = Fraction(2.435*pow(10, 18))
lam = pow(10, -3)
g = 1
q = 1
xi = Fraction((0.6*pow(10, 16)/MPL)**2)
xit = xi/(3*alp*q)

# --- functions ---
# Phi
def Phi(phi,s):
    return -3+s**2/2+(1+chi)*phi**2/2

# -Phi/3
def mPhiov3(phi,s):
    return Fraction(-1,3)*Phi(phi,s)

# Psi
def Psi(phi,s):
    return (mPhiov3(phi,s))**(2-3*alp)*lam**2/(2*alp**2*q*g**2*xi)*phi**2

# Kahler potenaial K
def K(phi, s):
    return -3*alp*sym.log(Fraction(-1,3)*Phi(phi, s))

# scalar potential V
def V(phi, s):
    VFE = (mPhiov3(phi, s))**(1-3*alp)*1/alp*lam**2/4*phi**2*s**2
    VDE = g**2/8*((mPhiov3(phi,s))**(-1)*alp*q*s**2-2*xi)**2
    return VFE + VDE

# dV/dphi
def dVovdphi(argphi,args):
    return sym.diff(V(phi,s),phi).subs([(phi,argphi),(s,args)])

# dV/ds
def dVovds(argphi,args):
    return sym.diff(V(phi,s),s).subs([(phi,argphi),(s,args)])

# d^2K/dphi^2
def Kpp(argphi, args):
    return sym.diff(K(phi, s), phi, 2).subs([(phi,argphi),(s,args)])

# d^2K/ds^2
def Kss(argphi, args):
    return sym.diff(K(phi, s), s, 2).subs([(phi,argphi),(s,args)])

# d(d^2K/dphi^2)/dphi
def Kppp(argphi, args):
    return sym.diff(K(phi, s), phi, 3).subs([(phi,argphi),(s,args)])

# d(d^2K/ds^2)/ds
def Ksss(argphi, args):
    return sym.diff(K(phi, s), s, 3).subs([(phi,argphi),(s,args)])

# apploximate local minimum s_min^2 
def s2appr(phi):
    return 2*3*xit*(mPhiov3(phi,0))*(1-Psi(phi,0))/(1+xit*(1-Psi(phi,0)))

# energy density
def rho(phi,s):
    return V(phi,s)

# Hubble parameter H
def H(phi,s):
    return sym.sqrt(rho(phi,s)*Fraction(1,3))

# --- initial conditions ---
# critical-point value
phic0sq = 2*alp**2*q*g**2*xi/(lam**2)
# deviation
delta = pow(10,-1)
# initial value of inflaton field
phi_init = sym.sqrt(phic0sq*(1-delta))
# initial value of waterfall field
s_init = sym.sqrt(s2appr(phi_init))
# initial value of dphi/dt
phid_init = 0
# initial value of ds/dt
sd_init = -dVovds(phi_init,s_init)/(Ksss(phi_init,s_init)*3*H(phi_init,s_init))


# --- main ---
dphiovdt = sym.lambdify((t,phi,s),-dVovdphi(phi,s)/(3*H(phi,s)*Kpp(phi,s)),"scipy")
dsovdt = sym.lambdify((t,phi,s),-dVovds(phi,s)/(3*H(phi,s)*Kss(phi,s)),"scipy")

if (__name__ == '__main__'):
    def rhss(t, fields):
        phi,s = fields
        dphidt = dphiovdt(t,phi,s)
        dsdt = dsovdt(t,phi,s)

        return [dphidt, dsdt]

    # --- initial conditins ---
    init = [phi_init,s_init]

    t_min = 0
    t_max = 8.517*10**6
    t_span = [t_min, t_max]
    n_t = 10**4
    t_eval = np.linspace(t_min, t_max, n_t)

    sol = solve_ivp(rhss, t_span, init, method='Radau', dense_output=True)
    sol_sm = sol.sol(t_eval)
    # print(sol.y)
    # print(sol_sm)

    # --- plot ---
    N=100
    fig, ax = plt.subplots()
    phi_list = np.linspace(0, float(phi_init),N)
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$ s$')
    ax.set_title('trajectory')
    ax.grid(color='gray', linestyle=':')
    ax.plot(sol_sm[0,:], sol_sm[1,:], color="tab:green",label="num(sm)")
    ax.plot(sol.y[0,:], sol.y[1,:], color="tab:blue", linestyle='-.', label="num")
    ax.plot(phi_list,[sym.sqrt(s2appr(phi_list[k])) for k in range(N)],color="tab:orange",linestyle=":",label="app")
    ax.legend(loc=0)
    fig.tight_layout()

    pdf = PdfPages('figures/trajectory.pdf')
    pdf.savefig(fig)
    pdf.close()

    plt.show()

