from numpy import uint
from pssolver import SpectralSolver
from pssolver.utils import fft, visualize1D

import torch
from tqdm import trange


N = 256
L = 256
dt = 0.04
steps = 10000
batchsize = 6

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
solver = SpectralSolver(shape=(N,), L=L, dt=dt, device=device, batchsize= batchsize )

# # --- Parameters (Constants) ---
k = 1

# # --- Add active fields ---
u_init = torch.zeros(batchsize, N)
centers = torch.randint(0, N, (batchsize,))
width = N // 20  # Adjust the width of the Gaussian

x = torch.arange(N)
for i in range(batchsize):
    u_init[i] = 100 * torch.exp(-0.5 * ((x - centers[i]) / width) ** 2)

solver.model.add_dynamic_field(
    "u",
    init = u_init,
    L_hat = -k*solver.q2
)

solver.build()

traj = []
for i in trange(steps):
    solver.run(1)
    if i % (steps//100) == 0:
        traj.append(solver.fields['u'])

traj = torch.stack(traj).permute(1,0,2)  # permutation: (num_snapshots, batch_size, N) --> (batch_size, num_snapshots, N)

print(f"Simulation finished. Data has shape: {traj.shape}")
visualize1D(data = traj.cpu().numpy(), filename="diffusion1D.png")

