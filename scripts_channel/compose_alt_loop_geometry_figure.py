#!/usr/bin/env python3
"""Compose the alternate three-panel loop view above the middle-channel geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]


def fit_image(
    image: Image.Image,
    size: tuple[int, int],
    *,
    horizontal_align: str = "center",
    vertical_align: str = "center",
) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    if horizontal_align == "left":
        x = 0
    elif horizontal_align == "right":
        x = size[0] - image.width
    else:
        x = (size[0] - image.width) // 2
    if vertical_align == "top":
        y = 0
    elif vertical_align == "bottom":
        y = size[1] - image.height
    else:
        y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def content_bbox(image: Image.Image, *, threshold: int = 245) -> tuple[int, int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    mask = np.any(arr < threshold, axis=2)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, 0, image.width, image.height)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def loop_panel_frame_span(image: Image.Image) -> tuple[int, int]:
    arr = np.asarray(image.convert("RGB"))
    # The upper part contains the time arrow. Restrict to the panel region so
    # the detected span follows the local-box frames.
    mask = np.any(arr[120:, :, :] < 245, axis=2)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, image.width)
    return (int(xs.min()), int(xs.max()) + 1)


def paste_clipped(canvas: Image.Image, image: Image.Image, xy: tuple[int, int]) -> None:
    x, y = xy
    src_left = max(0, -x)
    src_top = max(0, -y)
    src_right = min(image.width, canvas.width - x)
    src_bottom = min(image.height, canvas.height - y)
    if src_left >= src_right or src_top >= src_bottom:
        return
    cropped = image.crop((src_left, src_top, src_right, src_bottom))
    canvas.paste(cropped, (max(x, 0), max(y, 0)))


def find_red_bbox(
    image: Image.Image,
    region: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = region
    arr = np.asarray(image.convert("RGB"))
    crop = arr[y0:y1, x0:x1, :]
    mask = (crop[:, :, 0] > 170) & (crop[:, :, 1] < 90) & (crop[:, :, 2] < 90)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (
        x0 + int(xs.min()),
        y0 + int(ys.min()),
        x0 + int(xs.max()) + 1,
        y0 + int(ys.max()) + 1,
    )


def find_loop_panel_boxes(
    image: Image.Image,
    *,
    origin: tuple[int, int],
    source_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    origin_x, origin_y = origin
    source_width, source_height = source_size
    arr = np.asarray(image.convert("RGB"))
    boxes = []
    for index in range(3):
        x0 = max(0, round(origin_x + index * source_width / 3))
        x1 = min(image.width, round(origin_x + (index + 1) * source_width / 3))
        y0 = max(0, origin_y + 120)
        y1 = min(image.height, origin_y + min(source_height, round(source_height * 0.88)))
        if x0 >= x1 or y0 >= y1:
            continue
        crop = arr[y0:y1, x0:x1, :]
        mask = np.any(crop < 245, axis=2)
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        padding = 18
        boxes.append(
            (
                max(0, x0 + int(xs.min()) - padding),
                max(0, y0 + int(ys.min()) - padding),
                min(image.width - 1, x0 + int(xs.max()) + padding),
                min(image.height - 1, y0 + int(ys.max()) + padding),
            )
        )
    return boxes


def add_zoom_callouts(
    canvas: Image.Image,
    *,
    loop_origin: tuple[int, int],
    loop_source_size: tuple[int, int],
    geometry_origin: tuple[int, int],
    geometry_size: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    red = (210, 35, 35)
    line_width = 4
    geometry_x, geometry_y = geometry_origin
    geometry_width, geometry_height = geometry_size

    source_box = find_red_bbox(
        canvas,
        (
            geometry_x,
            geometry_y,
            geometry_x + geometry_width,
            geometry_y + geometry_height,
        ),
    )
    if source_box is None:
        return

    panel_boxes = find_loop_panel_boxes(
        canvas,
        origin=loop_origin,
        source_size=loop_source_size,
    )
    source_top = ((source_box[0] + source_box[2]) // 2, source_box[1])
    for box in panel_boxes:
        target = ((box[0] + box[2]) // 2, box[3])
        draw.line((source_top, target), fill=red, width=3)

    draw.rectangle(source_box, outline=red, width=line_width)
    for box in panel_boxes:
        draw.rectangle(box, outline=red, width=line_width)


def compose(loop_panel_path: Path, geometry_path: Path, output_path: Path) -> None:
    loop_panel = Image.open(loop_panel_path).convert("RGB")
    geometry = Image.open(geometry_path).convert("RGB")

    margin = 36
    row_gap = 18
    geom_left, _, geom_right, _ = content_bbox(geometry)
    loop_left, loop_right = loop_panel_frame_span(loop_panel)

    geometry_width = geometry.width
    geometry_height = geometry.height
    target_width = geom_right - geom_left
    source_width = loop_right - loop_left
    scale = target_width / source_width
    loop_width = round(loop_panel.width * scale)
    loop_height = round(loop_panel.height * scale)
    loop_panel = loop_panel.resize((loop_width, loop_height), Image.Resampling.LANCZOS)

    width = geometry_width + 2 * margin
    total_height = margin + loop_height + row_gap + geometry_height + margin

    canvas = Image.new("RGB", (width, total_height), "white")
    geometry_x = margin
    loop_x = round(geometry_x + geom_left - loop_left * scale)

    paste_clipped(canvas, loop_panel, (loop_x, margin))
    geometry_y = margin + loop_height + row_gap
    canvas.paste(geometry, (geometry_x, geometry_y))
    add_zoom_callouts(
        canvas,
        loop_origin=(loop_x, margin),
        loop_source_size=loop_panel.size,
        geometry_origin=(geometry_x, geometry_y),
        geometry_size=geometry.size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"png={output_path}")


def parse_args() -> argparse.Namespace:
    default_dir = REPO_ROOT / "data_control_box"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loop-panel",
        type=Path,
        default=default_dir / "loop_growth_local_clean" / "loop_growth_3panel_alt.png",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=default_dir / "mid_channel_loop_geometry" / "mid_channel_loop_geometry.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "mid_channel_loop_geometry" / "loop_alt_geometry_summary.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compose(args.loop_panel, args.geometry, args.output)


if __name__ == "__main__":
    main()
