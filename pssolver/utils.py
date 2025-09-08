import torch

def fft(field: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the batched N-dimensional Fast Fourier Transform (FFT) of the input tensor along the last `dim` dimensions.

    Args:
        field (torch.Tensor): Input tensor containing the data to transform.
        dim (int): Number of dimensions (from the end) over which to compute the FFT.

    Returns:
        torch.Tensor: The FFT of the input tensor, with the same device as `field`.
    """
    return torch.fft.fftn(field, dim=tuple(range(-dim, 0)))


def ifft(field: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the batched N-dimensional Inverse Fast Fourier Transform (IFFT) of the input tensor along the last `dim` dimensions and returns the real part.

    Args:
        field (torch.Tensor): Input tensor containing the data to inverse transform.
        dim (int): Number of dimensions (from the end) over which to compute the IFFT.

    Returns:
        torch.Tensor: The real part of the IFFT of the input tensor, with the same device as `field`.
    """
    return torch.fft.ifftn(field, dim=tuple(range(-dim, 0))).real


def visualize2D(data, filename="output.mp4", fps=60, cmap="viridis"):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np

    batchsize = data.shape[0]

    ncols = min(batchsize, 3)
    nrows = (batchsize + 2) // 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if batchsize == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    ims = []
    for i in range(batchsize):
        im = axes[i].imshow(data[i][0], cmap=cmap, origin='lower')
        plt.colorbar(im, ax=axes[i])
        ims.append(im)

    def update(frame):
        for i in range(batchsize):
            ims[i].set_data(data[i][frame])
            ims[i].set_clim(vmin=0, vmax=1)
        fig.canvas.draw_idle()
        return ims

    ani = animation.FuncAnimation(
        fig, update, frames=data.shape[1], interval=1000/fps, blit=True
    )
    ani.save(filename, writer='ffmpeg', fps=fps)
    plt.show()
    plt.close(fig)
        
        
def visualize1D(data, filename="output.png", cmap="viridis"):
    import matplotlib.pyplot as plt
    import numpy as np

    # data: shape [batchsize, n_frames, n_points]
    batchsize, n_frames, n_points = data.shape

    ncols = min(batchsize, 3)
    nrows = (batchsize + 2) // 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if batchsize == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(batchsize):
        im = axes[i].imshow(data[i].T, aspect='auto', cmap=cmap, origin='lower')
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Spatial Position")
        plt.colorbar(im, ax=axes[i], label="Value")
        axes[i].set_title(f"Batch {i+1}")

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close(fig)
        

# def visualize_pygame(self, data, scale=2, cmap="viridis"):
#     import pygame
#     import matplotlib.pyplot as plt
#     import numpy as np

#     pygame.init()
#     n_frames, N, _ = data.shape
#     width, height = N * scale, N * scale

#     # Prepare colormap
#     cmap_func = plt.get_cmap(cmap)
#     norm = lambda arr: (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

#     screen = pygame.display.set_mode((width, height))
#     pygame.display.set_caption("PDE Visualization (Interactive)")

#     running = True
#     frame = 0
#     clock = pygame.time.Clock()

#     while running:
#         keys = pygame.key.get_pressed()
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_ESCAPE:
#                     running = False

#         # Move frames while holding arrow keys
#         if keys[pygame.K_RIGHT]:
#             frame = min(frame + 1, n_frames - 1)
#         if keys[pygame.K_LEFT]:
#             frame = max(frame - 1, 0)

#         arr = data[frame].cpu().numpy()
#         arr_norm = norm(arr)
#         arr_rgb = (cmap_func(arr_norm)[..., :3] * 255).astype(np.uint8)
#         surf = pygame.surfarray.make_surface(np.transpose(np.kron(arr_rgb, np.ones((scale, scale, 1))), (1, 0, 2)))
#         screen.blit(surf, (0, 0))
#         pygame.display.flip()
#         clock.tick(30)

#     pygame.quit()