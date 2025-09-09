# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
from pssolver import SpectralSolver
from pssolver.utils import fft, visualize2D
from tqdm import trange


def Q_init(shape, seed = 42):
    Nx,Ny = shape
    generator = torch.Generator().manual_seed(seed)
    angle_radians = torch.tensor(torch.pi / 3)
    nx_ = torch.cos(angle_radians)
    ny_ = torch.sin(angle_radians)
    Qxx = (nx_ * nx_ - 0.5) + 0.01*(2 *torch.rand((Nx, Ny), generator=generator) - 1)
    Qxy = nx_ * ny_ + 0.01*(2 * torch.rand((Nx, Ny), generator=generator) - 1)

    return Qxx , Qxy
     

class NonlinearModel(torch.nn.Module):
    def __init__(self, solver):
        super().__init__()
        self.qx = solver.qx
        self.qy = solver.qy
        self.q2 = solver.q2
        
    def forward(self, fields, params): 
        Qxx = fields['Qxx']  
        Qxy = fields['Qxy']
        Qsq = Qxx**2 + Qxy**2

        ux = fields['ux']
        uy = fields['uy']
        w  = fields['w']
        gxQxx = fields['gradx_Qxx']
        gyQxx = fields['grady_Qxx']
        gxQxy = fields['gradx_Qxy']
        gyQxy = fields['grady_Qxy']
        Axx = fields['Axx']
        Axy = fields['Axy']

        lam = 1
        out0 =  -a4*Qsq*Qxx - ux*gxQxx - uy*gyQxx - 2*Qxy*w + lam*Axx  
        out1 =  -a4*Qsq*Qxy - ux*gxQxy - uy*gyQxy + 2*Qxx*w + lam*Axy  
        
        return torch.fft.fft2(torch.stack([out0, out1]))  

class Static_compute_fn(torch.nn.Module):
    def __init__(self, solver):
        super().__init__()
        qx = solver.qx
        qy = solver.qy
        q2 = solver.q2
        batchsize = solver.batchsize

        iqx = 1j * qx
        iqy = 1j * qy
        self.iqx = iqx
        self.iqy = iqy

        P = torch.zeros((2, 2, batchsize, *q2.shape), dtype=torch.cfloat, device=q2.device)
        P[0, 0] = 1 - (qx * qx) / q2
        P[0, 1] = - (qx * qy) / q2
        P[1, 0] = - (qy * qx) / q2
        P[1, 1] = 1 - (qy * qy) / q2
        self.P = P * 1/(fric+eta*q2)

        sig_to_f = torch.zeros((2, 2, batchsize, *q2.shape), dtype=torch.cfloat, device=q2.device)
        sig_to_f[0, 0] = iqx
        sig_to_f[0, 1] = iqy
        sig_to_f[1, 0] = -iqy
        sig_to_f[1, 1] = iqx
        self.sig_to_f = sig_to_f

    ### avoid repeating same calculations
    def forward(self, fields, params): 
        Qxx = fields['Qxx']  
        Qxy = fields['Qxy']
        alpha = params['alpha']
        Q = torch.stack([Qxx, Qxy], dim=0) # shape -> (2,B, *shape)
        sig =  beta * alpha * Q  
        sig_hat = torch.fft.fft2(sig)

        f_hat = torch.einsum('ijBxy,jBxy->iBxy', self.sig_to_f, sig_hat)  
        u_hat = torch.einsum('ijBxy,jBxy->iBxy', self.P, f_hat)

        # u_hat = 0*u_hat
        ux_hat = u_hat[0]
        uy_hat = u_hat[1]
        w_hat = 0.5*(self.iqx*uy_hat - self.iqy*ux_hat)  

        Axx_hat = self.iqx * ux_hat
        Axy_hat = 0.5*(self.iqx*uy_hat + self.iqy*ux_hat)
        
        Qxx_hat = fields['Qxx.hat']
        Qxy_hat = fields['Qxy.hat']   
        gradx_Qxx_hat = self.iqx * Qxx_hat
        grady_Qxx_hat = self.iqy * Qxx_hat
        gradx_Qxy_hat = self.iqx * Qxy_hat
        grady_Qxy_hat = self.iqy * Qxy_hat
        # return order is [ux, uy, w, Axx, Axy, gxQxx, gyQxx, gxQxy, gyQxy]
        return torch.stack([
            ux_hat,
            uy_hat,
            w_hat,
            Axx_hat,
            Axy_hat,
            gradx_Qxx_hat,
            grady_Qxx_hat,
            gradx_Qxy_hat,
            grady_Qxy_hat
        ])




seed = 24
N = 128
L = 256
dt = 0.01
steps = 40000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batchsize = 5

solver = SpectralSolver(shape = (N,N), L=L, dt=dt, device=device, batchsize= batchsize)

Qxx_0, Qxy_0 = Q_init(shape = (N,N), seed = seed)

# # --- Parameters ---
a2 = -1
a4 = 1
KQ = 10

beta = -1

# --- Add active fields ---
solver.model.add_dynamic_field(
    "Qxx",
    init = Qxx_0,
    L_hat = -( a2 + solver.q2 * KQ)
)
solver.model.add_dynamic_field(
    "Qxy",
    init =  Qxy_0,
    L_hat = -( a2 + solver.q2 * KQ)
)

fric = 0.01
eta = 1
# --- Add static fields ---
solver.model.add_static_field("ux")
solver.model.add_static_field("uy")
solver.model.add_static_field("w")
solver.model.add_static_field("Axx")
solver.model.add_static_field("Axy")
solver.model.add_static_field("gradx_Qxx")
solver.model.add_static_field("grady_Qxx")
solver.model.add_static_field("gradx_Qxy")
solver.model.add_static_field("grady_Qxy")

compiled_nl_model = torch.compile(NonlinearModel(solver),  mode="max-autotune")
compiled_static_model = torch.compile(Static_compute_fn(solver), mode="max-autotune")
solver.model.set_nonlinear_model(compiled_nl_model)
solver.model.set_static_compute_model(compiled_static_model)


alpha = torch.stack([0.2*(i+1)*torch.ones(( N, N), device=device) for i in range(batchsize)])
# alpha = 0.4 * torch.ones((batchsize, N, N), device=device)
solver.parameters.new_param('alpha', alpha)

solver.build()
print(solver.fields.name_to_idx)

traj = []
for i in trange(steps):
    if i % (steps//100) == 0:
        snapshot = torch.stack([solver.fields[name].clone() for name in ["Qxx", "Qxy"]]) # shape -> (2, batch, N,N)
        traj.append(snapshot)
    if i==steps//2:
        solver.parameters['alpha'] = torch.zeros((batchsize, N, N), device=device)
    solver.run(1)

    
traj = torch.stack(traj) # shape -> (time, 2, batch, N,N)
traj = traj.permute(2,1,0,3,4) # shape -> (batch, 2, time, N,N)


qxx = traj[:, 0]  # shape: (B, time, nx, ny)
qxy = traj[:, 1]  # shape: (B, time, nx, ny)

# Calculate scalar order parameter s
s = torch.sqrt(qxx**2 + qxy**2)  # shape: (B, time, nx, ny)

visualize2D(data = s.cpu().numpy())
