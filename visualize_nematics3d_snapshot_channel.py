#!/usr/bin/env python3
"""Visualize one channel-geometry nematic snapshot using Nematics3D."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE = REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
LOCAL_NEMATICS_SRC_CANDIDATES = [
    REPO_ROOT / "Nematics3D" / "src",
    REPO_ROOT.parent / "Develop" / "Nematics3D" / "src",
]

for local_nematics_src in LOCAL_NEMATICS_SRC_CANDIDATES:
    if local_nematics_src.exists():
        sys.path.insert(0, str(local_nematics_src))
        break

# ---------------- User-editable visualization configuration ----------------
# Edit values in this block when tuning channel figures. The script is fixed to
# the solver's channel snapshot layout: x is the streamwise periodic direction,
# while y and z are wall-bounded transverse directions.

DATA_DIR = Path(__file__).resolve().parent / "data_control_box_Nx256_Lx64"  # Directory containing Q_<index>.npy and u_<index>.npy.
SNAPSHOT_INDEX = 4000  # Reads DATA_DIR / f"Q_{SNAPSHOT_INDEX}.npy" and DATA_DIR / f"u_{SNAPSHOT_INDEX}.npy".
OUTPUT_DIR = DATA_DIR / "nematics3d_channel_figures"  # Directory where channel PNG files are written.
DEFECTS_PNG = None  # Defect-line PNG path; None uses OUTPUT_DIR / "<snapshot>_defects_lines.png".
VELOCITY_PNG = None  # Velocity-vector PNG path; None uses OUTPUT_DIR / "<snapshot>_velocity_vectors.png".
PLOT_DEFECTS = True  # If True, generate the defect/disclination-line figure when defects are present.
PLOT_VELOCITY = True  # If True, generate the velocity-vector figure.
SKIP_EMPTY_DEFECT_FIGURE = True  # If True, do not save a blank defect figure when no defects are detected.

DEFECT_THRESHOLD = 0.0  # Threshold passed to nematics3d.defect_detect.
PERIODIC_BOUNDARY = (True, False, False)  # Channel boundary flags: x periodic, y/z wall-bounded.
DETECTION_PLANES = (True, True, True)  # Which local planes nematics3d checks for defects: xy, xz, yz.
REQUIRE_CHANNEL_GRID = True  # If True, reject non-channel domains such as square plane snapshots.
PLOT_ACTIVITY_BOX = True  # Draw the local activity box used by Channel_control_box.py.
ACTIVITY_BOX_BOUNDS = ((118, 138), (4, 36), (4, 36))  # Half-open x, y, z grid bounds.
ACTIVITY_BOX_RADIUS = 0.22
ACTIVITY_BOX_COLOR = (0.85, 0.10, 0.10)
ACTIVITY_BOX_OPACITY = 0.85
PLOT_OBSERVATION_BOX = True  # Draw the larger box used for local defect counting.
OBSERVATION_BOX_BOUNDS = ((114, 142), (0, 40), (0, 40))
OBSERVATION_BOX_RADIUS = 0.13
OBSERVATION_BOX_COLOR = (0.95, 0.55, 0.05)
OBSERVATION_BOX_OPACITY = 0.60

# ---------------- Defect-line figure configuration ----------------
DEFECT_FIGURE_SIZE = (2600, 600)  # Defect figure size in pixels.
DEFECT_SAVE_SCALE = 3  # Defect screenshot scale factor.
DEFECT_BACKGROUND_COLOR = (1.0, 1.0, 1.0)  # Defect figure background RGB color, values in [0, 1].
DEFECT_CAMERA = "xy"  # Defect figure camera view: "isometric", "xy", "xz", or "yz".
DEFECT_FOCAL_POINT_OFFSET = (0.0, 0.0, 0.0)  # Defect figure camera target offset from the box center in x, y, z.
DEFECT_CAMERA_DISTANCE = 280  # Defect figure camera distance; None keeps Nematics3D's default camera distance.
DEFECT_AZIMUTH = 90  # Defect figure camera azimuth in degrees; None keeps the selected camera view value.
DEFECT_ELEVATION = None  # Defect figure camera elevation in degrees; None keeps the selected camera view value.
DEFECT_ROLL = None  # Defect figure camera roll in degrees; None keeps the selected camera view value.
DEFECT_DRAW_BOX = True  # If True, draw the simulation-domain bounding box in the defect figure.
DEFECT_BOX_RADIUS = 0.16  # Tube radius for the defect-figure bounding box.
DEFECT_BOX_COLOR = (0.12, 0.12, 0.12)  # RGB color of the defect-figure bounding box.
DEFECT_BOX_OPACITY = 0.45  # Opacity of the defect-figure bounding box.
DEFECT_BOX_SIDES = 12  # Number of sides in the defect-figure box tubes.

HIDE_DEFECT_POINTS = True  # If True, hide raw defect points and show only classified lines.
DEFECT_POINT_RADIUS = 0.28  # Sphere radius for raw defect points when they are shown.
DEFECT_POINT_COLOR = (0.82, 0.06, 0.16)  # RGB color for raw defect points.
DEFECT_POINT_OPACITY = 0.45  # Opacity for raw defect points; 1 is opaque, 0 is transparent.

SMOOTH_LINES = False  # If True, use Nematics3D line smoothing before plotting disclination lines.
SMOOTH_WINDOW_LENGTH = 3  # Savitzky-Golay smoothing window length for disclination lines.
SMOOTH_N_OUT_RATIO = 0.5  # Output point density after smoothing; larger values make curves visually smoother.
MIN_LINE_LENGTH = 2  # Minimum disclination-line point count for smoothing.
WRAP_LINES_FOR_DISPLAY = True  # If True, wrap periodic disclination lines into the primary simulation box for display.
CLIP_LINES_TO_BOX = True  # If True, use Nematics3D Bounds clipping so plotted lines do not extend outside the box.
LINE_CLIP_MARGIN = 1.5  # Inward display-only clipping margin in grid units; helps hide tube-radius protrusion at box faces.
LINE_CLIP_MODE = "mesh"  # Nematics3D tube clipping mode: "mesh" clips tube surfaces; "center" clips centerline points.
LINE_RADIUS = 0.24  # Tube radius for plotted disclination lines.
LINE_COLOR = (0.0, 0.0, 0.0)  # RGB color for plotted disclination lines.
LINE_OPACITY = 1.0  # Opacity of plotted disclination lines.
LINE_SIDES = 24  # Number of sides in disclination-line tubes; larger values make rounder tubes.

# ---------------- Velocity-vector figure configuration ----------------
VELOCITY_FIGURE_SIZE = (2600, 600)  # Velocity figure size in pixels.
VELOCITY_SAVE_SCALE = 3  # Velocity screenshot scale factor.
VELOCITY_BACKGROUND_COLOR = (1.0, 1.0, 1.0)  # Velocity figure background RGB color, values in [0, 1].
VELOCITY_CAMERA = "xy"  # Streamwise-wall-normal channel slice view: "isometric", "xy", "xz", or "yz".
VELOCITY_FOCAL_POINT_OFFSET = (0.0, 0.0, 0.0)  # Velocity figure camera target offset from the box center in x, y, z.
VELOCITY_CAMERA_DISTANCE = 280  # Velocity figure camera distance; None keeps Nematics3D's default camera distance.
VELOCITY_AZIMUTH = None  # Velocity figure camera azimuth in degrees; None keeps the selected camera view value.
VELOCITY_ELEVATION = None  # Velocity figure camera elevation in degrees; None keeps the selected camera view value.
VELOCITY_ROLL = 90  # Rotate the xy velocity view so streamwise x is horizontal in the saved figure.
VELOCITY_DRAW_BOX = True  # If True, draw the simulation-domain bounding box in the velocity figure.
VELOCITY_BOX_RADIUS = 0.07  # Tube radius for the velocity-figure bounding box.
VELOCITY_BOX_COLOR = (0.18, 0.18, 0.18)  # RGB color of the velocity-figure bounding box.
VELOCITY_BOX_OPACITY = 0.32  # Opacity of the velocity-figure bounding box.
VELOCITY_BOX_SIDES = 10  # Number of sides in the velocity-figure box tubes.

VELOCITY_Z_INDEX = None  # Z slice used for streamwise xy velocity vectors; None uses the channel midplane.
VELOCITY_STRIDE = 8  # Grid spacing between sampled velocity vectors in x and y.
VELOCITY_LENGTH_MODE = "normalized"  # Arrow length mode: "normalized", "constant", or "raw_scaled".
VELOCITY_CONSTANT_LENGTH = 6.0  # Nematics3D PlotVector length when VELOCITY_LENGTH_MODE == "constant".
VELOCITY_RAW_LENGTH_SCALE = 700.0  # Multiplier for raw |u| when VELOCITY_LENGTH_MODE == "raw_scaled".
VELOCITY_MIN_LENGTH = 3.0  # Display length for the smallest nonzero velocity vector.
VELOCITY_MAX_LENGTH = 13.5  # Display length for the largest velocity vector in the plotted slice.
VELOCITY_RADIUS = 0.30  # Shaft radius/thickness for velocity arrows.
VELOCITY_TIP_LENGTH_FRACTION = 0.36  # Fraction of each velocity arrow used by the arrow head.
VELOCITY_TIP_RADIUS_RATIO = 2.7  # Arrow-head radius divided by shaft radius.
VELOCITY_VECTOR_SIDES = 16  # Number of sides in velocity-arrow shafts and tips.
VELOCITY_CMAP = "viridis"  # Colormap used to color velocity arrows by speed magnitude.
VELOCITY_SCALAR_BAR_TITLE = "|u|"  # Title shown on the velocity-magnitude colorbar.
VELOCITY_SHOW_SCALAR_BAR = True  # If True, draw Nematics3D's in-figure scalar bar.
VELOCITY_SCALAR_BAR_N_LABELS = 5  # Number of labels shown on the velocity colorbar.
VELOCITY_SCALAR_BAR_POSITION = (0.940, 0.28)  # Colorbar lower-left corner in normalized figure coordinates.
VELOCITY_SCALAR_BAR_SIZE = (0.035, 0.48)  # Colorbar width and height in normalized figure coordinates.
VELOCITY_SCALAR_BAR_FONT_SIZE = 30  # Label font size for the velocity colorbar.
VELOCITY_SHOW_PLANE_LABEL = True  # If True, add a concise plane label to the velocity figure.
VELOCITY_PLANE_LABEL_POSITION = (0.035, 0.90)  # Viewport coordinates for the velocity plane label.
VELOCITY_PLANE_LABEL_FONT_SIZE = 20  # Font size for the velocity plane label.

PLOT_VORTICITY_BACKGROUND = True  # If True, draw channel midplane omega_z = d uy/dx - d ux/dy behind arrows.
VORTICITY_SURFACE_STRIDE = 2  # Grid spacing used for the vorticity PlotSurface background.
VORTICITY_SURFACE_Z_OFFSET = -0.15  # Small display offset so arrows sit above the scalar surface.
VORTICITY_SURFACE_OPACITY = 0.58  # Opacity of the omega_z scalar background.
VORTICITY_CMAP = "RdBu_r"  # Diverging colormap used for signed omega_z.
VORTICITY_CLIM_PERCENTILE = 99.0  # Symmetric color limit percentile for robust omega_z contrast.
VORTICITY_SCALAR_BAR_TITLE = ""  # Native VTK scalar-bar title; Unicode omega is added with a text overlay.
VORTICITY_SCALAR_BAR_LABEL = "ω_z"  # Display label shown above the vorticity colorbar.
VORTICITY_SCALAR_BAR_N_LABELS = 5  # Number of labels shown on the vorticity colorbar.
VORTICITY_SCALAR_BAR_POSITION = (0.015, 0.28)  # Vorticity colorbar lower-left corner in normalized coordinates.
VORTICITY_SCALAR_BAR_SIZE = (0.035, 0.48)  # Vorticity colorbar width and height in normalized coordinates.
VORTICITY_SCALAR_BAR_FONT_SIZE = 28  # Label font size for the vorticity colorbar.
VORTICITY_SCALAR_BAR_LABEL_POSITION = (0.024, 0.81)  # Viewport coordinates for the vorticity colorbar label.
VORTICITY_SCALAR_BAR_LABEL_FONT_SIZE = 18  # Font size for the vorticity colorbar label.
VORTICITY_SCALAR_BAR_LABEL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Font file for Unicode omega.
# ---------------------------------------------------------------------------

CHANNEL_PERIODIC_BOUNDARY = (True, False, False)
CHANNEL_LONG_AXIS = 0

if VENV_SITE.exists():
    sys.path.append(str(VENV_SITE))

try:
    import nematics3d as n3d
    from nematics3d.classes.grid_field import GridFieldDataset, InputGridField
    from nematics3d.classes.bounds import OptsBounds
    from nematics3d.classes.vector_plane import VectorPlane
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "nematics3d with PlotVector/VectorPlane support is required. Run with:\n"
        "  /home/fansenwei/anaconda3/envs/Nematics3D/bin/python "
        "PSSolver/visualize_nematics3d_snapshot_channel.py\n"
        "and keep /home/fansenwei/Desktop/Develop/Nematics3D/src available."
    ) from exc


def require_component_last(
    array: np.ndarray,
    components: int,
    name: str,
    path: Path,
) -> np.ndarray:
    """Validate the solver data-folder component-last array format."""
    if array.ndim == 4 and array.shape[-1] == components:
        return np.asarray(array, dtype=np.float64)
    raise ValueError(
        f"{path} has unexpected {name} shape {array.shape}; expected "
        f"(Nx, Ny, Nz, {components}) from the PSSolver channel snapshot format"
    )


def snapshot_paths(index: int) -> tuple[Path, Path]:
    """Return the fixed PSSolver channel snapshot pair for an integer index."""
    if not isinstance(index, int):
        raise TypeError("SNAPSHOT_INDEX must be an integer")
    return DATA_DIR / f"Q_{index}.npy", DATA_DIR / f"u_{index}.npy"


def read_data_snapshot(index: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Read one Q/u snapshot from the channel data folder's fixed file layout."""
    q_path, u_path = snapshot_paths(index)
    if not q_path.exists():
        raise FileNotFoundError(f"Q snapshot does not exist: {q_path}")
    if not u_path.exists():
        raise FileNotFoundError(f"velocity snapshot does not exist: {u_path}")

    q5 = require_component_last(np.load(q_path), 5, "Q", q_path)
    velocity = require_component_last(np.load(u_path), 3, "velocity", u_path)
    if q5.shape[:3] != velocity.shape[:3]:
        raise ValueError(
            f"Q and velocity grid shapes do not match: {q_path} has {q5.shape[:3]}, "
            f"{u_path} has {velocity.shape[:3]}"
        )
    return q5, velocity, q_path.stem


def validate_channel_snapshot(shape: tuple[int, int, int]) -> None:
    """Reject obvious non-channel grids before producing misleading figures."""
    nx, ny, nz = shape
    if tuple(bool(v) for v in PERIODIC_BOUNDARY) != CHANNEL_PERIODIC_BOUNDARY:
        raise ValueError(
            "visualize_nematics3d_snapshot_channel.py is channel-only: "
            f"PERIODIC_BOUNDARY must be {CHANNEL_PERIODIC_BOUNDARY} for x-periodic, y/z-wall geometry"
        )
    if REQUIRE_CHANNEL_GRID and nx <= max(ny, nz):
        raise ValueError(
            "visualize_nematics3d_snapshot_channel.py expects a streamwise channel grid "
            f"with Nx larger than the transverse dimensions; got shape {shape}"
        )


def detect_lines(
    q5: np.ndarray,
    threshold: float,
    periodic: tuple[bool, bool, bool],
    planes: tuple[bool, bool, bool],
) -> tuple[np.ndarray, list]:
    _, director = n3d.Q_diagonalize(q5)
    defect_indices = n3d.defect_detect(
        director,
        threshold=threshold,
        is_boundary_periodic=periodic,
        planes=planes,
    )
    box_size_periodic = [
        director.shape[i] if periodic[i] else np.inf
        for i in range(3)
    ]
    lines = n3d.defect_classify_into_lines(
        defect_indices,
        box_size_periodic=box_size_periodic,
        grid_offset=np.zeros(3),
        grid_transform=np.eye(3),
    )
    return defect_indices, lines


def box_center(shape: tuple[int, int, int]) -> tuple[float, float, float]:
    nx, ny, nz = shape
    return (nx / 2, ny / 2, nz / 2)


def offset_focal_point(
    shape: tuple[int, int, int],
    offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(np.asarray(box_center(shape), dtype=float) + np.asarray(offset, dtype=float))


def domain_bounds(shape: tuple[int, int, int], margin: float = 0.0) -> n3d.Bounds:
    nx, ny, nz = shape
    return n3d.Bounds(
        opts=OptsBounds(
            origin=(margin, margin, margin),
            axis1=(1.0, 0.0, 0.0),
            axis2=(0.0, 1.0, 0.0),
            length1=nx - 2.0 * margin,
            length2=ny - 2.0 * margin,
            length3=nz - 2.0 * margin,
            alignment="min_corner",
        )
    )


def make_figure(
    shape: tuple[int, int, int],
    name: str,
    figure_size: tuple[int, int],
    background_color: tuple[float, float, float],
    draw_box: bool,
    box_radius: float,
    box_color: tuple[float, float, float],
    box_opacity: float,
    box_sides: int,
) -> n3d.PlotFigure:
    nx, ny, nz = shape
    fig = n3d.PlotFigure(
        is_off_screen=True,
        name=name,
        opts=n3d.OptsFigure(
            size=figure_size,
            bg_color=background_color,
            focal_point=box_center(shape),
        ),
    )
    if draw_box:
        n3d.PlotExtent(
            n3d.get_box_corners(nx, ny, nz),
            figure=fig,
            opts=n3d.OptsTube(
                radius=box_radius,
                color=box_color,
                opacity=box_opacity,
                sides=box_sides,
                is_scalar_bar=False,
                is_reset_camera=False,
            ),
        )
    for enabled, bounds, radius, color, opacity, label in (
        (
            PLOT_OBSERVATION_BOX,
            OBSERVATION_BOX_BOUNDS,
            OBSERVATION_BOX_RADIUS,
            OBSERVATION_BOX_COLOR,
            OBSERVATION_BOX_OPACITY,
            "observation box",
        ),
        (
            PLOT_ACTIVITY_BOX,
            ACTIVITY_BOX_BOUNDS,
            ACTIVITY_BOX_RADIUS,
            ACTIVITY_BOX_COLOR,
            ACTIVITY_BOX_OPACITY,
            "activity box",
        ),
    ):
        if enabled:
            corners = n3d.get_box_corners(
                bounds[0][1] - bounds[0][0],
                bounds[1][1] - bounds[1][0],
                bounds[2][1] - bounds[2][0],
            )
            corners += np.array(
                [bounds[0][0], bounds[1][0], bounds[2][0]],
                dtype=float,
            )
            n3d.PlotExtent(
                corners,
                figure=fig,
                name=label,
                opts=n3d.OptsTube(
                    radius=radius,
                    color=color,
                    opacity=opacity,
                    sides=12,
                    is_scalar_bar=False,
                    is_reset_camera=False,
                ),
            )
    return fig


def set_camera(
    fig: n3d.PlotFigure,
    camera: str,
    focal_point: tuple[float, float, float],
    distance: float | None,
    azimuth: float | None,
    elevation: float | None,
    roll: float | None,
) -> None:
    if camera == "xy":
        fig.act_view_xy()
    elif camera == "xz":
        fig.act_view_xz()
    elif camera == "yz":
        fig.act_view_yz()
    else:
        fig.act_view_isometric()
    fig.act_commit(focal_point=focal_point)
    if distance is not None:
        fig.act_commit(distance=distance)
    camera_updates = {
        key: value
        for key, value in {
            "azimuth": azimuth,
            "elevation": elevation,
            "roll": roll,
        }.items()
        if value is not None
    }
    if camera_updates:
        fig.act_commit(**camera_updates)


def save_figure(fig: n3d.PlotFigure, output: Path, save_scale: int | float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.act_savefig(str(output), scale=save_scale)


def plot_defects_and_lines(
    output: Path,
    shape: tuple[int, int, int],
    defects: np.ndarray,
    lines: list,
) -> None:
    fig = make_figure(
        shape,
        "defects and disclination lines",
        DEFECT_FIGURE_SIZE,
        DEFECT_BACKGROUND_COLOR,
        DEFECT_DRAW_BOX,
        DEFECT_BOX_RADIUS,
        DEFECT_BOX_COLOR,
        DEFECT_BOX_OPACITY,
        DEFECT_BOX_SIDES,
    )
    bounds = domain_bounds(shape, LINE_CLIP_MARGIN) if CLIP_LINES_TO_BOX else None

    if not HIDE_DEFECT_POINTS and len(defects) > 0:
        n3d.PlotSphere(
            defects,
            figure=fig,
            opts=n3d.OptsSphere(
                radius=DEFECT_POINT_RADIUS,
                color=DEFECT_POINT_COLOR,
                opacity=DEFECT_POINT_OPACITY,
                is_scalar_bar=False,
                is_reset_camera=False,
            ),
        )

    tube_opts = n3d.OptsTube(
        radius=LINE_RADIUS,
        color=LINE_COLOR,
        opacity=LINE_OPACITY,
        sides=LINE_SIDES,
        is_scalar_bar=False,
        is_reset_camera=False,
    )
    for line in lines:
        if SMOOTH_LINES:
            line.act_smooth(
                window_length=SMOOTH_WINDOW_LENGTH,
                num_out_ratio=SMOOTH_N_OUT_RATIO,
                min_line_length=MIN_LINE_LENGTH,
            )
        line.act_visualize(
            figure=fig,
            is_wrap=WRAP_LINES_FOR_DISPLAY,
            is_smooth=SMOOTH_LINES,
            opts=tube_opts,
            bounds=bounds,
            clip_mode=LINE_CLIP_MODE,
        )

    set_camera(
        fig,
        DEFECT_CAMERA,
        offset_focal_point(shape, DEFECT_FOCAL_POINT_OFFSET),
        DEFECT_CAMERA_DISTANCE,
        DEFECT_AZIMUTH,
        DEFECT_ELEVATION,
        DEFECT_ROLL,
    )
    save_figure(fig, output, DEFECT_SAVE_SCALE)


def velocity_display_length(speed: np.ndarray) -> np.ndarray:
    speed = np.asarray(speed, dtype=float)
    speed_max = float(np.nanmax(speed))
    if speed_max <= 1e-12:
        return np.full_like(speed, VELOCITY_MIN_LENGTH)
    return VELOCITY_MIN_LENGTH + (VELOCITY_MAX_LENGTH - VELOCITY_MIN_LENGTH) * speed / speed_max


def velocity_length_setting():
    """Return a Nematics3D PlotVector length setting."""
    if VELOCITY_LENGTH_MODE == "constant":
        return VELOCITY_CONSTANT_LENGTH
    if VELOCITY_LENGTH_MODE == "raw_scaled":
        return lambda speed: VELOCITY_RAW_LENGTH_SCALE * np.asarray(speed, dtype=float)
    return velocity_display_length


def derivative_axis(field: np.ndarray, axis: int, periodic: bool) -> np.ndarray:
    """Return a unit-grid first derivative along one axis."""
    if periodic:
        return 0.5 * (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis))
    return np.gradient(field, axis=axis, edge_order=2)


def vorticity_z(velocity: np.ndarray) -> np.ndarray:
    """Return omega_z = d uy/dx - d ux/dy on the full grid."""
    ux = velocity[..., 0]
    uy = velocity[..., 1]
    d_uy_dx = derivative_axis(uy, axis=0, periodic=bool(PERIODIC_BOUNDARY[0]))
    d_ux_dy = derivative_axis(ux, axis=1, periodic=bool(PERIODIC_BOUNDARY[1]))
    return d_uy_dx - d_ux_dy


def symmetric_clim(values: np.ndarray, percentile: float) -> tuple[float, float]:
    """Return robust symmetric scalar limits for a signed field."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    scale = float(np.percentile(np.abs(finite), percentile))
    if scale <= 1e-14:
        scale = float(np.max(np.abs(finite)))
    if scale <= 1e-14:
        scale = 1.0
    return (-scale, scale)


def plot_vorticity_background(
    fig: n3d.PlotFigure,
    omega_z: np.ndarray,
    z_index: int,
) -> None:
    """Draw an omega_z scalar surface behind the velocity arrows."""
    if float(np.max(np.abs(omega_z))) <= 1e-12:
        return

    nx, ny = omega_z.shape
    stride = int(VORTICITY_SURFACE_STRIDE)
    x = np.arange(0, nx, stride, dtype=float)
    y = np.arange(0, ny, stride, dtype=float)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = np.full_like(xx, float(z_index) + VORTICITY_SURFACE_Z_OFFSET, dtype=float)
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    scalars = omega_z[::stride, ::stride].ravel()
    clim = symmetric_clim(scalars, VORTICITY_CLIM_PERCENTILE)

    surface = n3d.PlotSurface(
        coords,
        figure=fig,
        name="omega_z background",
        paint_by="scalars",
        scalars=scalars,
        scalars_cmap=VORTICITY_CMAP,
        scalars_clim=clim,
        scalar_bar_title=VORTICITY_SCALAR_BAR_TITLE,
        opacity=VORTICITY_SURFACE_OPACITY,
        is_scalar_bar=False,
        is_reset_camera=False,
    )
    fig.pl.add_scalar_bar(
        title=VORTICITY_SCALAR_BAR_TITLE,
        mapper=surface.entity_actor.mapper,
        n_labels=VORTICITY_SCALAR_BAR_N_LABELS,
        position_x=VORTICITY_SCALAR_BAR_POSITION[0],
        position_y=VORTICITY_SCALAR_BAR_POSITION[1],
        width=VORTICITY_SCALAR_BAR_SIZE[0],
        height=VORTICITY_SCALAR_BAR_SIZE[1],
        vertical=True,
        title_font_size=VORTICITY_SCALAR_BAR_FONT_SIZE,
        label_font_size=VORTICITY_SCALAR_BAR_FONT_SIZE,
        color="black",
        fmt="%.2g",
        outline=False,
        render=False,
    )
    fig.pl.add_text(
        VORTICITY_SCALAR_BAR_LABEL,
        position=VORTICITY_SCALAR_BAR_LABEL_POSITION,
        font_size=VORTICITY_SCALAR_BAR_LABEL_FONT_SIZE,
        color="black",
        viewport=True,
        font_file=VORTICITY_SCALAR_BAR_LABEL_FONT,
    )


def plot_velocity_vectors(
    output: Path,
    velocity: np.ndarray,
) -> None:
    nx, ny, nz, _ = velocity.shape
    speed_max = float(np.max(np.linalg.norm(velocity, axis=-1)))
    z_index = nz // 2 if VELOCITY_Z_INDEX is None else VELOCITY_Z_INDEX
    if not 0 <= z_index < nz:
        raise ValueError(f"VELOCITY_Z_INDEX={z_index} outside valid range [0, {nz - 1}]")

    dataset = GridFieldDataset(
        inputValue=InputGridField(
            shape=(nx, ny, nz),
            box_periodic_flag=tuple(bool(v) for v in PERIODIC_BOUNDARY),
        ),
        name="velocity dataset",
    )
    field = dataset.act_add_field("velocity", velocity)
    interpolator = field.act_add_interpolator()
    plane = VectorPlane(
        interpolator=interpolator,
        name=f"velocity z={z_index}",
        opts=n3d.OptsPlaneGrid(
            normal=(0.0, 0.0, 1.0),
            axis1=(1.0, 0.0, 0.0),
            origin=(0.0, 0.0, float(z_index)),
            alignment="bottom-left",
            spacing=float(VELOCITY_STRIDE),
            spacing_extra=float(VELOCITY_STRIDE),
            size=float(nx - 1),
            size_extra=float(ny - 1),
        ),
    )

    fig = make_figure(
        (nx, ny, nz),
        "velocity vectors",
        VELOCITY_FIGURE_SIZE,
        VELOCITY_BACKGROUND_COLOR,
        VELOCITY_DRAW_BOX,
        VELOCITY_BOX_RADIUS,
        VELOCITY_BOX_COLOR,
        VELOCITY_BOX_OPACITY,
        VELOCITY_BOX_SIDES,
    )
    if PLOT_VORTICITY_BACKGROUND:
        plot_vorticity_background(fig, vorticity_z(velocity)[:, :, z_index], z_index)

    if speed_max > 1e-12:
        vector_visual = plane.act_visualize_vector(
            figure=fig,
            resolver_source="orient_length",
            paint_by="scalars",
            scalars=lambda speed: speed,
            scalars_cmap=VELOCITY_CMAP,
            scalar_bar_title=VELOCITY_SCALAR_BAR_TITLE,
            length=velocity_length_setting(),
            radius=VELOCITY_RADIUS,
            tip_length_fraction=VELOCITY_TIP_LENGTH_FRACTION,
            tip_radius_ratio=VELOCITY_TIP_RADIUS_RATIO,
            anchor="center",
            sides=VELOCITY_VECTOR_SIDES,
            is_scalar_bar=False,
            is_reset_camera=False,
        )
        if VELOCITY_SHOW_SCALAR_BAR:
            fig.pl.add_scalar_bar(
                title=VELOCITY_SCALAR_BAR_TITLE,
                mapper=vector_visual.entity_actor.mapper,
                n_labels=VELOCITY_SCALAR_BAR_N_LABELS,
                position_x=VELOCITY_SCALAR_BAR_POSITION[0],
                position_y=VELOCITY_SCALAR_BAR_POSITION[1],
                width=VELOCITY_SCALAR_BAR_SIZE[0],
                height=VELOCITY_SCALAR_BAR_SIZE[1],
                vertical=True,
                title_font_size=VELOCITY_SCALAR_BAR_FONT_SIZE,
                label_font_size=VELOCITY_SCALAR_BAR_FONT_SIZE,
                color="black",
                fmt="%.2g",
                outline=False,
                render=False,
            )
    else:
        fig.pl.add_text(
            "velocity field is zero (alpha = 0)",
            position=(0.035, 0.82),
            font_size=VELOCITY_PLANE_LABEL_FONT_SIZE,
            color="black",
            viewport=True,
        )
    if VELOCITY_SHOW_PLANE_LABEL:
        fig.pl.add_text(
            f"channel xy midplane, z = {z_index}",
            position=VELOCITY_PLANE_LABEL_POSITION,
            font_size=VELOCITY_PLANE_LABEL_FONT_SIZE,
            color="black",
            viewport=True,
        )
    set_camera(
        fig,
        VELOCITY_CAMERA,
        offset_focal_point((nx, ny, nz), VELOCITY_FOCAL_POINT_OFFSET),
        VELOCITY_CAMERA_DISTANCE,
        VELOCITY_AZIMUTH,
        VELOCITY_ELEVATION,
        VELOCITY_ROLL,
    )
    save_figure(fig, output, VELOCITY_SAVE_SCALE)


def main() -> None:
    output_dir = OUTPUT_DIR

    valid_cameras = {"isometric", "xy", "xz", "yz"}
    if DEFECT_CAMERA not in valid_cameras:
        raise ValueError("DEFECT_CAMERA must be one of: isometric, xy, xz, yz")
    if VELOCITY_CAMERA not in valid_cameras:
        raise ValueError("VELOCITY_CAMERA must be one of: isometric, xy, xz, yz")
    if not PLOT_DEFECTS and not PLOT_VELOCITY:
        raise ValueError("At least one of PLOT_DEFECTS or PLOT_VELOCITY must be True")
    if len(PERIODIC_BOUNDARY) != 3 or len(DETECTION_PLANES) != 3:
        raise ValueError("PERIODIC_BOUNDARY and DETECTION_PLANES must each have three values")
    if VELOCITY_STRIDE < 1:
        raise ValueError("VELOCITY_STRIDE must be >= 1")
    if VORTICITY_SURFACE_STRIDE < 1:
        raise ValueError("VORTICITY_SURFACE_STRIDE must be >= 1")
    if VELOCITY_LENGTH_MODE not in {"normalized", "constant", "raw_scaled"}:
        raise ValueError('VELOCITY_LENGTH_MODE must be "normalized", "constant", or "raw_scaled"')
    if VELOCITY_CONSTANT_LENGTH <= 0:
        raise ValueError("VELOCITY_CONSTANT_LENGTH must be > 0")
    if VELOCITY_RAW_LENGTH_SCALE <= 0:
        raise ValueError("VELOCITY_RAW_LENGTH_SCALE must be > 0")
    if VELOCITY_MIN_LENGTH < 0:
        raise ValueError("VELOCITY_MIN_LENGTH must be >= 0")
    if VELOCITY_MAX_LENGTH <= VELOCITY_MIN_LENGTH:
        raise ValueError("VELOCITY_MAX_LENGTH must be > VELOCITY_MIN_LENGTH")
    if VELOCITY_VECTOR_SIDES < 3:
        raise ValueError("VELOCITY_VECTOR_SIDES must be >= 3")
    if not (0 < VELOCITY_TIP_LENGTH_FRACTION < 1):
        raise ValueError("VELOCITY_TIP_LENGTH_FRACTION must be in (0, 1)")
    if VELOCITY_TIP_RADIUS_RATIO <= 0:
        raise ValueError("VELOCITY_TIP_RADIUS_RATIO must be > 0")
    if VELOCITY_SCALAR_BAR_N_LABELS < 2:
        raise ValueError("VELOCITY_SCALAR_BAR_N_LABELS must be >= 2")
    if len(VELOCITY_SCALAR_BAR_POSITION) != 2:
        raise ValueError("VELOCITY_SCALAR_BAR_POSITION must contain two values")
    if len(VELOCITY_SCALAR_BAR_SIZE) != 2 or min(VELOCITY_SCALAR_BAR_SIZE) <= 0:
        raise ValueError("VELOCITY_SCALAR_BAR_SIZE must contain two positive values")
    if VELOCITY_SCALAR_BAR_FONT_SIZE <= 0:
        raise ValueError("VELOCITY_SCALAR_BAR_FONT_SIZE must be > 0")
    if VELOCITY_PLANE_LABEL_FONT_SIZE <= 0:
        raise ValueError("VELOCITY_PLANE_LABEL_FONT_SIZE must be > 0")
    if not (0 <= VORTICITY_SURFACE_OPACITY <= 1):
        raise ValueError("VORTICITY_SURFACE_OPACITY must be in [0, 1]")
    if not (0 < VORTICITY_CLIM_PERCENTILE <= 100):
        raise ValueError("VORTICITY_CLIM_PERCENTILE must be in (0, 100]")
    if len(VORTICITY_SCALAR_BAR_POSITION) != 2:
        raise ValueError("VORTICITY_SCALAR_BAR_POSITION must contain two values")
    if len(VORTICITY_SCALAR_BAR_SIZE) != 2 or min(VORTICITY_SCALAR_BAR_SIZE) <= 0:
        raise ValueError("VORTICITY_SCALAR_BAR_SIZE must contain two positive values")
    if VORTICITY_SCALAR_BAR_N_LABELS < 2:
        raise ValueError("VORTICITY_SCALAR_BAR_N_LABELS must be >= 2")
    if VORTICITY_SCALAR_BAR_FONT_SIZE <= 0:
        raise ValueError("VORTICITY_SCALAR_BAR_FONT_SIZE must be > 0")
    for name, value in {
        "DEFECT_CAMERA_DISTANCE": DEFECT_CAMERA_DISTANCE,
        "VELOCITY_CAMERA_DISTANCE": VELOCITY_CAMERA_DISTANCE,
    }.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be > 0 or None")
    for name, value in {
        "DEFECT_AZIMUTH": DEFECT_AZIMUTH,
        "VELOCITY_AZIMUTH": VELOCITY_AZIMUTH,
    }.items():
        if value is not None and not (0 <= value <= 360):
            raise ValueError(f"{name} must be in [0, 360] or None")
    for name, value in {
        "DEFECT_ELEVATION": DEFECT_ELEVATION,
        "VELOCITY_ELEVATION": VELOCITY_ELEVATION,
    }.items():
        if value is not None and not (-90 <= value <= 90):
            raise ValueError(f"{name} must be in [-90, 90] or None")
    for name, value in {
        "DEFECT_ROLL": DEFECT_ROLL,
        "VELOCITY_ROLL": VELOCITY_ROLL,
    }.items():
        if value is not None and not (-180 <= value <= 180):
            raise ValueError(f"{name} must be in [-180, 180] or None")
    for name, value in {
        "DEFECT_SAVE_SCALE": DEFECT_SAVE_SCALE,
        "VELOCITY_SAVE_SCALE": VELOCITY_SAVE_SCALE,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    for name, value in {
        "DEFECT_FIGURE_SIZE": DEFECT_FIGURE_SIZE,
        "VELOCITY_FIGURE_SIZE": VELOCITY_FIGURE_SIZE,
    }.items():
        if len(value) != 2 or min(value) <= 0:
            raise ValueError(f"{name} must contain two positive values")
    for name, value in {
        "DEFECT_BACKGROUND_COLOR": DEFECT_BACKGROUND_COLOR,
        "VELOCITY_BACKGROUND_COLOR": VELOCITY_BACKGROUND_COLOR,
        "DEFECT_BOX_COLOR": DEFECT_BOX_COLOR,
        "VELOCITY_BOX_COLOR": VELOCITY_BOX_COLOR,
        "DEFECT_POINT_COLOR": DEFECT_POINT_COLOR,
        "LINE_COLOR": LINE_COLOR,
    }.items():
        if len(value) != 3:
            raise ValueError(f"{name} must have three RGB values")
    for name, value in {
        "DEFECT_FOCAL_POINT_OFFSET": DEFECT_FOCAL_POINT_OFFSET,
        "VELOCITY_FOCAL_POINT_OFFSET": VELOCITY_FOCAL_POINT_OFFSET,
    }.items():
        if len(value) != 3:
            raise ValueError(f"{name} must have three values")
    if SMOOTH_WINDOW_LENGTH < 3:
        raise ValueError("SMOOTH_WINDOW_LENGTH must be >= 3")
    if SMOOTH_N_OUT_RATIO <= 0:
        raise ValueError("SMOOTH_N_OUT_RATIO must be > 0")
    if MIN_LINE_LENGTH < 2:
        raise ValueError("MIN_LINE_LENGTH must be >= 2")
    if DEFECT_BOX_SIDES < 3 or VELOCITY_BOX_SIDES < 3 or LINE_SIDES < 3:
        raise ValueError("DEFECT_BOX_SIDES, VELOCITY_BOX_SIDES, and LINE_SIDES must be >= 3")
    for name, value in {
        "DEFECT_BOX_RADIUS": DEFECT_BOX_RADIUS,
        "VELOCITY_BOX_RADIUS": VELOCITY_BOX_RADIUS,
        "LINE_RADIUS": LINE_RADIUS,
        "DEFECT_POINT_RADIUS": DEFECT_POINT_RADIUS,
        "VELOCITY_RADIUS": VELOCITY_RADIUS,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    for name, value in {
        "DEFECT_BOX_OPACITY": DEFECT_BOX_OPACITY,
        "VELOCITY_BOX_OPACITY": VELOCITY_BOX_OPACITY,
        "LINE_OPACITY": LINE_OPACITY,
        "DEFECT_POINT_OPACITY": DEFECT_POINT_OPACITY,
    }.items():
        if not (0 <= value <= 1):
            raise ValueError(f"{name} must be in [0, 1]")
    if LINE_CLIP_MODE not in {"mesh", "center"}:
        raise ValueError('LINE_CLIP_MODE must be either "mesh" or "center"')
    if LINE_CLIP_MARGIN < 0:
        raise ValueError("LINE_CLIP_MARGIN must be >= 0")

    q_path, u_path = snapshot_paths(SNAPSHOT_INDEX)
    q5, velocity, input_label = read_data_snapshot(SNAPSHOT_INDEX)
    validate_channel_snapshot(q5.shape[:3])
    defects_png = DEFECTS_PNG or output_dir / f"{input_label}_defects_lines.png"
    velocity_png = VELOCITY_PNG or output_dir / f"{input_label}_velocity_vectors.png"
    if CLIP_LINES_TO_BOX and 2.0 * LINE_CLIP_MARGIN >= min(q5.shape[:3]):
        raise ValueError("LINE_CLIP_MARGIN is too large for the snapshot domain")
    defects, lines = detect_lines(
        q5,
        DEFECT_THRESHOLD,
        tuple(bool(v) for v in PERIODIC_BOUNDARY),
        tuple(bool(v) for v in DETECTION_PLANES),
    )

    print(f"data_dir={DATA_DIR}")
    print(f"snapshot_index={SNAPSHOT_INDEX}")
    print(f"q_path={q_path}")
    print(f"u_path={u_path}")
    print("geometry=channel")
    print(f"periodic_boundary={PERIODIC_BOUNDARY}")
    print(f"grid_shape={q5.shape[:3]}")
    print(f"defect_points={len(defects)}")
    print(f"disclination_lines={len(lines)}")

    if PLOT_DEFECTS and (len(defects) > 0 or not SKIP_EMPTY_DEFECT_FIGURE):
        plot_defects_and_lines(
            defects_png,
            q5.shape[:3],
            defects,
            lines,
        )
        print(f"defects_png={defects_png}")
    elif PLOT_DEFECTS:
        if defects_png.exists():
            defects_png.unlink()
        print("defects_png=skipped_empty")
    else:
        print("defects_png=skipped")

    if PLOT_VELOCITY:
        plot_velocity_vectors(
            velocity_png,
            velocity,
        )
        print(f"velocity_png={velocity_png}")
    else:
        print("velocity_png=skipped")


if __name__ == "__main__":
    main()
