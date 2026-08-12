# Auto-install and import dlroms
try:
    from dlroms import *
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "git+https://github.com/NicolaRFranco/dlroms.git"])
    from dlroms import *
import numpy as np
import matplotlib.pyplot as plt
import gdown


mesh = fe.unitsquaremesh(40, 40)
Vh = fe.space(mesh, 'CG', 1, vector_valued = True)


gdown.download(id = "1jMzgW3fe0A1BRiX1O4jH0T7GGUqnqVar", output = "sidewalk.npz")


data = np.load("sidewalk.npz")
mu, u = data['mu'], data['u']


#Computation of the average vertical deformation at the top boundary

ns, nt, nh = u.shape
indexes = 1-fe.dofs(Vh)[:, 1]<1e-12   # indexes of the dofs located at the top edge
tindex = np.arange(0, 51, 10)         # indexes of the times t_1, ..., t_q

def Q(u):
  """Given a space-time solution u, returns [Q(u,t_1), ..., Q(u, t_q)].
  If multiple solutions are passed, the output is computed batchwise.

  Works both on numpy arrays and torch tensors."""

  U = u.reshape(-1, nt, nh)[:, :, indexes]
  vals = U.reshape(-1, nt, np.sum(indexes)//2, 2)[:, :, :, 1].mean(axis = -1)[:, tindex]
  return vals if len(U)>1 else vals[0]

def visualize(u):
  fe.animate(u,Vh,warp=True,axis=[-0.25, 1.25, -0.25, 1.25],T=5,dt=0.1)
