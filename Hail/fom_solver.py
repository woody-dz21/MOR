"""
FOM Solver Module
"""

import numpy as np
import matplotlib.pyplot as plt

# Auto-install and import dlroms
try:
    from dlroms import *
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "git+https://github.com/NicolaRFranco/dlroms.git"])
    from dlroms import *

from ufl_legacy import nabla_div
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from fenics import Constant, DirichletBC, nabla_grad, Identity, TrialFunction, TestFunction, dot, dx, ds, inner, assemble

clc()

# Define global mesh and function spaces
mesh = fe.unitsquaremesh(40, 40) # rescaling from [0cm,6cm]x[0cm,6cm] to [0,1]x[0,1]
Vh = fe.space(mesh, 'CG', 1, vector_valued = True)
Dh = fe.space(mesh, 'DG', 0)
clc()

# Boundary conditions
tol = 1e-5
def clamped_boundary(x, on_boundary):
    return on_boundary and x[1]<tol
bc = DirichletBC(Vh, Constant((0, 0)), clamped_boundary)

def assemble_affine_bases():
    """
    Offline affine base assembly using standard imports and no classes.
    """
    u = TrialFunction(Vh)
    v = TestFunction(Vh)

    # Indicator functions for top and bottom subdomains
    top_indicator = fe.interpolate(lambda x: 1.0 * (x[1] > 0.5), Dh)
    bot_indicator = fe.interpolate(lambda x: 1.0 * (x[1] <= 0.5), Dh)

    def epsilon_u(w):
        return 0.5 * (nabla_grad(w) + nabla_grad(w).T)

    def sigma_base(w, indicator):
        return indicator * (nabla_div(w) * Identity(2) + 2 * epsilon_u(w))

    # Base Stiffness Matrices
    a1 = inner(sigma_base(u, top_indicator), epsilon_u(v)) * dx
    a2 = inner(sigma_base(u, bot_indicator), epsilon_u(v)) * dx

    A1 = assemble(a1)
    A2 = assemble(a2)

    bc.zero(A1)
    bc.apply(A2)

    A1_mat = csr_matrix(A1.array())
    A2_mat = csr_matrix(A2.array())

    # Base Load Vectors
    f_grav = Constant((0, -1))
    L_grav = dot(f_grav, v) * dx
    F_grav = assemble(L_grav)
    bc.apply(F_grav)

    x0, delta = 0.5, 0.1

    T_x = fe.interpolate(lambda x: [1.0 * (x[1] > 0.99) * (np.abs(x[0] - x0) < (delta + tol)), 0.0], Vh)
    L_Tx = dot(T_x, v) * ds
    F_Tx = assemble(L_Tx)
    bc.apply(F_Tx)

    T_y = fe.interpolate(lambda x: [0.0, 1.0 * (x[1] > 0.99) * (np.abs(x[0] - x0) < (delta + tol))], Vh)
    L_Ty = dot(T_y, v) * ds
    F_Ty = assemble(L_Ty)
    bc.apply(F_Ty)

    return A1_mat, A2_mat, F_grav[:], F_Tx[:], F_Ty[:]

def assemble_FOM(mu):
    r, m, theta = mu
    x0, delta = 0.5, 0.1 # hailstone of diameter 1.2cm becomes of radius 0.1 in the rescaled metric

    # Auxiliary definitions
    lambda_ = fe.interpolate(lambda x: r*(x[1]>0.5)*1.0+1.0*(x[1]<=0.5), Dh)
    nu = fe.interpolate(lambda x: (x[1]>0.5)*r+1.0*(x[1]<=0.5), Dh)

    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)

    def sigma(u):
        return lambda_*nabla_div(u)*Identity(2) + 2*nu*epsilon(u)

    # Variational problem
    u = TrialFunction(Vh)
    v = TestFunction(Vh)
    f = Constant((0, -1))
    T = fe.interpolate(lambda x: [m*np.cos(theta-np.pi/2)*(x[1] > 0.99)*(np.abs(x[0]-x0)<(delta+tol)),
                                  m*np.sin(theta-np.pi/2)*(x[1] > 0.99)*(np.abs(x[0]-x0)<(delta+tol))], Vh)

    a = inner(sigma(u), epsilon(v))*dx
    L = dot(f, v)*dx + dot(T, v)*ds

    # Assembling and adjusting
    A = assemble(a)
    F = assemble(L)
    bc.apply(A)
    bc.apply(F)

    A = csr_matrix(A.array())
    F = F[:]
    
    return A, F

def FOMsolver(mu):
    A, F = assemble_FOM(mu)

    # Solving
    u = spsolve(A, F)
    clc()
    return u
