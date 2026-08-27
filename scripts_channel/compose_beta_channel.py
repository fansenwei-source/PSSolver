#!/usr/bin/env python3
"""Compose the three-step beta panel above the alpha-box channel geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]


def compose(triptych_path: Path, channel_path: Path, output: Path) -> None:
    triptych = Image.open(triptych_path).convert("RGB")
    channel_source = Image.open(channel_path).convert("RGB")

    canvas_width = triptych.width
    channel_width = round(canvas_width * 0.96)
    channel_height = round(
        channel_source.height * channel_width / channel_source.width
    )
    channel = channel_source.resize(
        (channel_width, channel_height),
        Image.Resampling.LANCZOS,
    )

    channel_array = np.asarray(channel)
    red_mask = (
        (channel_array[:, :, 0] > 120)
        & (channel_array[:, :, 1] < 90)
        & (channel_array[:, :, 2] < 90)
    )
    red_pixels = channel_array[red_mask]
    if len(red_pixels):
        bright = red_pixels[
            red_pixels[:, 0] >= np.percentile(red_pixels[:, 0], 70)
        ]
        line_color = tuple(int(value) for value in np.median(bright, axis=0))
    else:
        line_color = (201, 42, 42)

    margin_top = 25
    gap = 120
    margin_bottom = 55
    canvas_height = (
        margin_top + triptych.height + gap + channel.height + margin_bottom
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    top_y = margin_top
    channel_x = (canvas_width - channel.width) // 2
    channel_y = margin_top + triptych.height + gap
    canvas.paste(triptych, (0, top_y))
    canvas.paste(channel, (channel_x, channel_y))

    draw = ImageDraw.Draw(canvas)
    ys, xs = np.where(red_mask)
    if xs.size:
        red_box = (
            channel_x + int(xs.min()),
            channel_y + int(ys.min()),
            channel_x + int(xs.max()),
            channel_y + int(ys.max()),
        )
        anchor = ((red_box[0] + red_box[2]) // 2, red_box[1])
    else:
        anchor = (canvas_width // 2, channel_y + channel.height // 4)

    panel_centers = [
        round(canvas_width * 0.166),
        round(canvas_width * 0.500),
        round(canvas_width * 0.834),
    ]
    line_y = top_y + triptych.height - 145
    for center_x in panel_centers:
        draw.line(
            [(center_x, line_y), anchor],
            fill=line_color,
            width=8,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"png={output}")


def parse_args() -> argparse.Namespace:
    default_dir = REPO_ROOT / "data_control_box" / "beta_channel"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--triptych",
        type=Path,
        default=default_dir / "beta_3steps.png",
    )
    parser.add_argument(
        "--channel",
        type=Path,
        default=default_dir / "channel_geometry.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "beta_3steps_channel.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compose(args.triptych, args.channel, args.output)


if __name__ == "__main__":
    main()
