import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def _require_pygame():
    """Load the optional display backend only when a display is requested."""
    try:
        import pygame
    except ImportError as error:
        raise ImportError(
            "Pygame visualization requires the optional dependency; "
            "install pssolver[visualization]."
        ) from error
    return pygame

class Visualizer:
    def __init__(self, width=800, height=600, title="PSS Visualizer"):
        self.width = width
        self.height = height
        self.title = title
        self.running = False

    def run_pygame(self, draw_callback=None, fps=60):
        pygame = _require_pygame()
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        clock = pygame.time.Clock()
        self.running = True

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            if draw_callback:
                draw_callback(screen)
            pygame.display.flip()
            clock.tick(fps)
        pygame.quit()

    def plot_static(self, data, cmap="viridis", colorbar=True, extent=None, title=None):
        plt.figure(figsize=(8, 6))
        arr = data.cpu().numpy() if hasattr(data, "cpu") else data
        im = plt.imshow(arr, cmap=cmap, origin='lower', extent=extent)
        if colorbar:
            plt.colorbar(im)
        if title:
            plt.title(title)
        plt.show()

    def animate_matplotlib(self, data, filename=None, fps=20, cmap="viridis", extent=None):
        arr = data.cpu().numpy() if hasattr(data, "cpu") else data
        fig, ax = plt.subplots()
        im = ax.imshow(arr[0], cmap=cmap, origin='lower', extent=extent)
        plt.colorbar(im, ax=ax)

        def update(frame):
            im.set_data(arr[frame])
            im.set_clim(vmin=arr[frame].min(), vmax=arr[frame].max())
            return [im]

        ani = animation.FuncAnimation(
            fig, update, frames=arr.shape[0], interval=1000/fps, blit=True
        )
        if filename:
            ani.save(filename, writer='ffmpeg', fps=fps)
        plt.show()
        plt.close(fig)

    def animate_pygame(self, data, scale=2, cmap="viridis"):
        pygame = _require_pygame()
        arr = data.cpu().numpy() if hasattr(data, "cpu") else data
        n_frames, N, _ = arr.shape
        width, height = N * scale, N * scale

        # Prepare colormap
        cmap_func = plt.get_cmap(cmap)
        def norm(a): return (a - a.min()) / (a.max() - a.min() + 1e-8)

        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(self.title)
        clock = pygame.time.Clock()
        running = True
        frame = 0

        while running:
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if keys[pygame.K_RIGHT]:
                frame = min(frame + 1, n_frames - 1)
            if keys[pygame.K_LEFT]:
                frame = max(frame - 1, 0)

            arr_norm = norm(arr[frame])
            arr_rgb = (cmap_func(arr_norm)[..., :3] * 255).astype(np.uint8)
            surf = pygame.surfarray.make_surface(
                np.transpose(np.kron(arr_rgb, np.ones((scale, scale, 1))), (1, 0, 2))
            )
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
        pygame.quit()

    def visualize(self, data, mode="matplotlib", **kwargs):
        """
        mode: 'matplotlib', 'matplotlib_anim', 'pygame_anim'
        kwargs: passed to the respective method
        """
        if mode == "matplotlib":
            self.plot_static(data, **kwargs)
        elif mode == "matplotlib_anim":
            self.animate_matplotlib(data, **kwargs)
        elif mode == "pygame_anim":
            self.animate_pygame(data, **kwargs)
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")
