#!/usr/bin/env python3
"""Compose the three beta snapshot panels used by the channel summary figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEPS = (6200, 6360, 7000)


def compose(input_dir: Path, output: Path, steps: tuple[int, ...]) -> None:
    images = [
        Image.open(input_dir / f"beta_{step}.png").convert("RGB")
        for step in steps
    ]

    # This crop preserves the complete translucent boxes and excludes the
    # colorbar from the third panel, matching the retained summary figure.
    common_crop = (1160, 0, 5100, 4500)
    panels = [image.crop(common_crop) for image in images]

    target_height = 1550
    resized = []
    for panel in panels:
        width, height = panel.size
        new_width = round(width * target_height / height)
        resized.append(
            panel.resize((new_width, target_height), Image.Resampling.LANCZOS)
        )

    panel_gap = 92
    margin_x = 40
    margin_y = 80
    canvas_width = margin_x * 2 + sum(image.width for image in resized)
    canvas_width += panel_gap * (len(resized) - 1)
    canvas_height = target_height + margin_y * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    first_arrow_shift = -18
    x = margin_x
    for index, panel in enumerate(resized):
        canvas.paste(panel, (x, margin_y))
        x += panel.width
        if index >= len(resized) - 1:
            continue

        y = canvas_height // 2
        center_x = x + panel_gap / 2
        if index == 0:
            center_x += first_arrow_shift
        arrow_length = 82
        head_length = 34
        head_half_height = 22
        tip_x = center_x + arrow_length / 2
        tail_x = center_x - arrow_length / 2
        base_x = tip_x - head_length
        draw.line(
            [(tail_x, y), (base_x, y)],
            fill=(25, 25, 25),
            width=16,
        )
        draw.polygon(
            [
                (tip_x, y),
                (base_x, y - head_half_height),
                (base_x, y + head_half_height),
            ],
            fill=(25, 25, 25),
        )
        x += panel_gap

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"png={output}")


def parse_args() -> argparse.Namespace:
    default_dir = REPO_ROOT / "data_control_box" / "beta_channel"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_dir)
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "beta_3steps.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compose(args.input_dir, args.output, tuple(int(step) for step in args.steps))


if __name__ == "__main__":
    main()
