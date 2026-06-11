import torch

def fft(field: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the batched periodic N-dimensional FFT along the last `dim` dimensions.

    For mixed boundary conditions, prefer `fields.transform_tensor(...)` or
    `solver.transform_tensor(...)` so each axis can choose FFT/DST/DCT as needed.

    Args:
        field (torch.Tensor): Input tensor containing the data to transform.
        dim (int): Number of dimensions (from the end) over which to compute the FFT.

    Returns:
        torch.Tensor: The FFT of the input tensor, with the same device as `field`.
    """
    return torch.fft.fftn(field, dim=tuple(range(-dim, 0)))


def ifft(field: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the batched periodic N-dimensional inverse FFT along the last
    `dim` dimensions and returns the real part.

    For mixed boundary conditions, prefer `fields.inverse_transform_tensor(...)`
    or `solver.inverse_transform_tensor(...)`.

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
