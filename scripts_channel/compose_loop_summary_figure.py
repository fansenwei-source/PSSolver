#!/usr/bin/env python3
"""Compose loop snapshots, alpha geometry, and radius history into one figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]


def trim_white_border(
    image: Image.Image,
    *,
    threshold: int = 248,
    padding: int = 8,
) -> Image.Image:
    image = image.convert("RGB")
    mask = image.point(lambda value: 255 if value < threshold else 0).convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        return image
    left, upper, right, lower = bbox
    left = max(0, left - padding)
    upper = max(0, upper - padding)
    right = min(image.width, right + padding)
    lower = min(image.height, lower + padding)
    return image.crop((left, upper, right, lower))


def dark_content_bbox(image: Image.Image, threshold: int = 80) -> tuple[int, int, int, int]:
    image = image.convert("RGB")
    mask = image.point(lambda value: 255 if value < threshold else 0).convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        return (0, 0, image.width, image.height)
    return bbox


def fit_image(
    image: Image.Image,
    size: tuple[int, int],
    *,
    width_scale: float = 1.0,
    horizontal_align: str = "center",
    vertical_align: str = "center",
) -> Image.Image:
    image = image.convert("RGB")
    if width_scale == 1.0:
        image.thumbnail(size, Image.Resampling.LANCZOS)
    else:
        target_width = int(size[0] * width_scale)
        target_height = round(image.height * target_width / image.width)
        if target_height > size[1]:
            target_height = size[1]
            target_width = round(image.width * target_height / image.height)
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
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


def find_red_bbox(
    image: Image.Image,
    region: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = region
    pixels = image.load()
    coords = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            r, g, b = pixels[x, y]
            if r > 170 and g < 80 and b < 80:
                coords.append((x, y))
    if not coords:
        return None
    xs = [item[0] for item in coords]
    ys = [item[1] for item in coords]
    return min(xs), min(ys), max(xs), max(ys)


def find_loop_panel_boxes(
    image: Image.Image,
    *,
    origin: tuple[int, int],
    source_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    origin_x, origin_y = origin
    source_width, source_height = source_size
    pixels = image.load()
    boxes = []
    for index in range(3):
        x0 = origin_x + int(index * source_width / 3)
        x1 = origin_x + int((index + 1) * source_width / 3)
        y0 = origin_y + 70
        y1 = origin_y + min(source_height, 410)
        coords = []
        for y in range(y0, y1):
            for x in range(x0, x1):
                r, g, b = pixels[x, y]
                if min(r, g, b) < 248:
                    coords.append((x, y))
        if not coords:
            continue
        xs = [item[0] for item in coords]
        ys = [item[1] for item in coords]
        padding = 16
        boxes.append(
            (
                max(0, min(xs) - padding),
                max(0, min(ys) - padding),
                min(image.width - 1, max(xs) + padding),
                min(image.height - 1, max(ys) + padding),
            )
        )
    return boxes


def add_zoom_callouts(
    canvas: Image.Image,
    *,
    margin: int,
    loop_width: int,
    loop_source_size: tuple[int, int],
    top_height: int,
    row_gap: int,
    geometry_height: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    red = (210, 35, 35)
    line_width = 4

    geometry_y = margin + top_height + row_gap
    source_box = find_red_bbox(
        canvas,
        (
            round(canvas.width * 0.40),
            geometry_y,
            round(canvas.width * 0.58),
            geometry_y + geometry_height,
        ),
    )
    if source_box is None:
        return

    panel_boxes = find_loop_panel_boxes(
        canvas,
        origin=(margin, margin),
        source_size=loop_source_size,
    )

    source_top = ((source_box[0] + source_box[2]) // 2, source_box[1])
    for box in panel_boxes:
        target = ((box[0] + box[2]) // 2, box[3])
        draw.line((source_top, target), fill=red, width=3)

    for box in panel_boxes:
        draw.rectangle(box, outline=red, width=line_width)


def restore_loop_panel_axes(
    canvas: Image.Image,
    top_left: Image.Image,
    *,
    origin: tuple[int, int],
) -> None:
    origin_x, origin_y = origin
    pixels = top_left.load()
    coords = []
    for y in range(round(top_left.height * 0.72), top_left.height):
        for x in range(round(top_left.width * 0.82), top_left.width):
            r, g, b = pixels[x, y]
            is_axis_color = (
                (r > 180 and g < 100 and b < 100)
                or (g > 120 and r < 120 and b < 150)
                or (b > 150 and r < 150 and g < 180)
            )
            if is_axis_color:
                coords.append((x, y))
    if not coords:
        return
    xs = [item[0] for item in coords]
    ys = [item[1] for item in coords]
    padding = 14
    left = max(0, min(xs) - padding)
    upper = max(0, min(ys) - padding)
    right = min(top_left.width, max(xs) + padding)
    lower = min(top_left.height, max(ys) + padding)
    axes = top_left.crop((left, upper, right, lower))
    canvas_pixels = canvas.load()
    axes_pixels = axes.load()
    for local_y in range(axes.height):
        for local_x in range(axes.width):
            r, g, b = axes_pixels[local_x, local_y]
            if min(r, g, b) < 245:
                canvas_pixels[origin_x + left + local_x, origin_y + upper + local_y] = (
                    255,
                    255,
                    255,
                )
    canvas.paste(axes, (origin_x + left + 46, origin_y + upper))


def compose(
    loop_panel_path: Path,
    geometry_path: Path,
    radius_path: Path,
    output_path: Path,
) -> None:
    loop_panel = trim_white_border(Image.open(loop_panel_path), padding=6)
    geometry = trim_white_border(Image.open(geometry_path), padding=6)
    radius = trim_white_border(Image.open(radius_path), padding=6)

    width = 2200
    margin = 36
    column_gap = 30
    row_gap = 20
    top_height = 520
    geometry_height = 360
    content_width = width - 2 * margin
    loop_width = 1340
    radius_width = content_width - loop_width - column_gap
    total_height = margin + top_height + row_gap + geometry_height + margin

    canvas = Image.new("RGB", (width, total_height), "white")
    top_left = fit_image(
        loop_panel,
        (loop_width, top_height),
        horizontal_align="left",
        vertical_align="top",
    )
    top_right = fit_image(
        radius,
        (radius_width, top_height),
        horizontal_align="right",
        vertical_align="top",
    )
    bottom = fit_image(
        geometry,
        (content_width, geometry_height),
        horizontal_align="center",
        vertical_align="top",
    )

    canvas.paste(top_left, (margin, margin))
    canvas.paste(top_right, (margin + loop_width + column_gap, margin))
    canvas.paste(bottom, (margin, margin + top_height + row_gap))
    add_zoom_callouts(
        canvas,
        margin=margin,
        loop_width=loop_width,
        loop_source_size=loop_panel.size,
        top_height=top_height,
        row_gap=row_gap,
        geometry_height=geometry_height,
    )
    restore_loop_panel_axes(canvas, top_left, origin=(margin, margin))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"png={output_path}")


def parse_args() -> argparse.Namespace:
    default_dir = REPO_ROOT / "data_control_box"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--five-panel",
        type=Path,
        default=default_dir / "loop_growth_local_clean" / "loop_growth_3panel.png",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=default_dir / "alpha_field_geometry" / "alpha_field_geometry.png",
    )
    parser.add_argument(
        "--radius",
        type=Path,
        default=default_dir / "loop_growth_local_clean" / "loop_radius_control_cycle.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "loop_growth_summary" / "loop_growth_summary.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compose(args.five_panel, args.geometry, args.radius, args.output)


if __name__ == "__main__":
    main()
