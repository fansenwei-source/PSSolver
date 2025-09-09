from pssolver import SpectralSolver
from pssolver.utils import fft
import torch
from tqdm import trange


class CH_NLmodel(torch.nn.Module):
    def forward(self, fields,params): 
        u = fields['u']  
        q2 = fields.q2

        output = - q2 * b * fft(u**3, dim=3)
        return output.unsqueeze(0)


N = 256
L = 128
dt = 0.1
steps = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
solver = SpectralSolver(shape=(N,N,N), L=L, dt=dt, device=device)

# # --- Parameters ---
a = -2
b = 1
k = 4

# # --- Add active fields ---
solver.model.add_dynamic_field(
    "u",
    init = 0.1 * torch.randn((N, N, N)),
    L_hat = -solver.q2 * (a + k*solver.q2)
)

solver.model.set_nonlinear_model(CH_NLmodel())
solver.build()

traj = []

for i in trange(steps, desc="Solving"):
    solver.run(1)
    if i % (steps//10) == 0:
        traj.append(solver.fields['u'].cpu())

traj = torch.stack(traj).permute(1,0,2,3,4)  # permutation: (num_snapshots, batch_size, N, N, N) --> (batch_size, num_snapshots, N, N, N)

print(traj.shape)
