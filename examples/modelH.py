import time
import torch
from pssolver import SpectralSolver
from pssolver.utils import fft, visualize2D

class NonlinearModel(torch.nn.Module):
    def forward(self, fields, params): 
        phi = fields['phi']  
        q2 = fields.q2
        
        ux = fields['ux']
        uy = fields['uy']
        gxphi = fields['gradx_phi']
        gyphi = fields['grady_phi']
       
        out0  =  - b* q2* fft(phi**3, dim=2) + fft(- ux*gxphi - uy*gyphi, dim=2)
        
        return out0.unsqueeze(0)

class Static_compute_fn(torch.nn.Module):
    def forward(self, fields, params): 
        ### avoid repeating same calculations

        # Cache coefficients and linear term on first forward call
        if not hasattr(self, 'P'):
            qx = fields.qx
            qy = fields.qy
            iqx = 1j * qx
            iqy = 1j * qy
            q2 = fields.q2
            self.iqx = iqx
            self.iqy = iqy

            batchsize = solver.batchsize

            # Stokes flow projection operator (in Fourier space)
            # Projects a vector field onto its divergence-free component
            P = torch.zeros((2, 2, batchsize, *q2.shape), dtype=torch.cfloat, device=q2.device)
            P[0, 0] = 1 - (qx * qx) / q2
            P[0, 1] = - (qx * qy) / q2
            P[1, 0] = - (qy * qx) / q2
            P[1, 1] = 1 - (qy * qy) / q2
            self.P = P * 1/(eta*q2)

            sig_to_f = torch.zeros((2, batchsize, *q2.shape), dtype=torch.cfloat, device=q2.device)
            sig_to_f[0] = self.iqx
            sig_to_f[1] = self.iqy
            self.sig_to_f = sig_to_f



        sigxx =  -k/2 * (fields['gradx_phi']**2 - fields['grady_phi']**2)  
        sigxy =  -k * (fields['gradx_phi'] * fields['grady_phi'])  
        sigxx_hat, sigxy_hat = torch.fft.fft2(torch.stack([sigxx,sigxy]))

        # Create sig tensor with shape (2,2, B,N,N)
        sig_hat = torch.zeros((2,2, *sigxx.shape), dtype=sigxy_hat.dtype, device = sigxy_hat.device)
        sig_hat[0,0] = sigxx_hat
        sig_hat[0,1] = sigxy_hat
        sig_hat[1,0] = sigxy_hat
        sig_hat[1,1] = -sigxx_hat

        f_hat = torch.einsum('jBxy,ijBxy->iBxy', self.sig_to_f, sig_hat)  
        u_hat = torch.einsum('ijBxy,jBxy->iBxy', self.P, f_hat) 

        ux_hat = u_hat[0]
        uy_hat = u_hat[1]
        
        phi_hat = fields['phi.hat']
        gradx_phi_hat = self.iqx * phi_hat
        grady_phi_hat = self.iqy * phi_hat

        # return order is [ux, uy, gxphi, gyphi]
        return torch.stack([
            ux_hat,
            uy_hat,
            gradx_phi_hat,
            grady_phi_hat,
        ])




seed = 42
N = 128
L = 256
dt = 0.1
steps = 20000
batchsize = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

solver = SpectralSolver(shape = (N,N), L=L, dt=dt, device=device, batchsize=batchsize)

# # --- Parameters ---
a = -1
b = 1
k = 4
eta = 1

# --- Add active fields ---
solver.model.add_dynamic_field(
    "phi",
    init =  0.1 * torch.randn((batchsize, N, N)),
    L_hat = - solver.q2*(a + k*solver.q2)
)

# --- Add static fields ---
solver.model.add_static_field("ux")
solver.model.add_static_field("uy")
solver.model.add_static_field("gradx_phi")
solver.model.add_static_field("grady_phi")


solver.model.set_nonlinear_model(NonlinearModel())
solver.model.set_static_compute_model(Static_compute_fn())

solver.build()
print(solver.fields.dyn_count)
print(solver.fields.name_to_idx)

traj = []
for i in range(steps):
    if i % (steps//100)== 0:
        snapshot = solver.fields['phi'] 
        traj.append(snapshot)
    solver.run(1)

traj = torch.stack(traj).permute(1,0,2,3) # permutation: (num_snapshots, batch_size, N, N) --> (batch_size, num_snapshots, N, N)

visualize2D(data = traj.cpu().numpy())
