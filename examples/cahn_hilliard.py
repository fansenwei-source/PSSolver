from pssolver import SpectralSolver
from pssolver.utils import fft, visualize2D

import torch
from tqdm import trange


class CH_NLmodel(torch.nn.Module):
    def forward(self, fields, params): 
        u = fields['u']  
        q2 = fields.q2
        b = params['b']

        output = - q2 * b * fft(u**3, dim=2)
        return output.unsqueeze(0)


N = 256
L = 256
dt = 0.1
steps = 100000
batchsize = 9

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
solver = SpectralSolver(shape=(N,N), L=L, dt=dt, device=device, batchsize= batchsize )

# # --- Parameters (Constants) ---
a = -2
b = 1
k = 4

# # --- Add active fields ---
solver.model.add_dynamic_field(
    "u",
    init = 0.1* torch.randn(batchsize, N, N),
    L_hat = -solver.q2 * (a + k*solver.q2)
)

solver.model.set_nonlinear_model(CH_NLmodel())
solver.parameters.new_param('b', torch.tensor(b, device=device))  # Parameter that can be accessed and modified during runtime
solver.build()

traj = []
for i in trange(steps):
    solver.run(1)
    if i % (steps//100) == 0:
        traj.append(solver.fields['u'])

traj = torch.stack(traj).permute(1,0,2,3)  # permutation: (num_snapshots, batch_size, N, N) --> (batch_size, num_snapshots, N, N)

print(traj.shape)
visualize2D(data = traj.cpu().numpy())

