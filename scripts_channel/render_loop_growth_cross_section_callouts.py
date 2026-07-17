#!/usr/bin/env python3
"""Render loop cross sections with in-place callouts and front-view enlargements."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import render_loop_growth_cross_section_front_views as front
import render_loop_growth_local_cross_sections as base


REPO_ROOT = Path(__file__).resolve().parents[1]


def fit_on_white_with_offset(path: Path, size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    canvas.paste(image, offset)
    return canvas, offset


def colored_director_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    arr = np.asarray(image.convert("RGB"))
    red = (arr[:, :, 0] > 130) & (arr[:, :, 1] < 95) & (arr[:, :, 2] < 95)
    blue = (arr[:, :, 2] > 110) & (arr[:, :, 0] < 120) & (arr[:, :, 1] < 140)
    mask = red | blue
    ys, xs = np.where(mask)
    if xs.size < 10:
        return []

    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    centers = np.array(
        [
            pts[np.argmin(pts[:, 0])],
            pts[np.argmax(pts[:, 0])],
        ],
        dtype=float,
    )
    for _ in range(12):
        dist = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for label in (0, 1):
            if np.any(labels == label):
                new_centers[label] = pts[labels == label].mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    boxes = []
    for label in (0, 1):
        cluster = pts[labels == label]
        if len(cluster) < 5:
            continue
        pad = 2
        x0 = max(0, int(np.floor(cluster[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(cluster[:, 1].min())) - pad)
        x1 = min(image.width - 1, int(np.ceil(cluster[:, 0].max())) + pad)
        y1 = min(image.height - 1, int(np.ceil(cluster[:, 1].max())) + pad)
        boxes.append((x0, y0, x1, y1))
    return sorted(boxes, key=lambda box: (box[0] + box[2]) / 2)


def red_point_anchors(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    arr = np.asarray(image.convert("RGB"))
    red = (arr[:, :, 0] > 130) & (arr[:, :, 1] < 95) & (arr[:, :, 2] < 95)
    anchors = []
    for box in boxes:
        x0, y0, x1, y1 = box
        crop = red[y0 : y1 + 1, x0 : x1 + 1]
        seen = np.zeros_like(crop, dtype=bool)
        candidates = []
        height, width = crop.shape
        for y in range(height):
            for x in range(width):
                if not crop[y, x] or seen[y, x]:
                    continue
                stack = [(y, x)]
                seen[y, x] = True
                xs = []
                ys = []
                while stack:
                    cy, cx = stack.pop()
                    xs.append(cx)
                    ys.append(cy)
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < height and 0 <= nx < width and crop[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                area = len(xs)
                comp_w = max(xs) - min(xs) + 1
                comp_h = max(ys) - min(ys) + 1
                aspect = max(comp_w / comp_h, comp_h / comp_w)
                if area >= 8 and comp_w >= 3 and comp_h >= 3 and aspect <= 2.2:
                    candidates.append((area, x0 + np.mean(xs), y0 + np.mean(ys)))
        if candidates:
            _, cx, cy = max(candidates, key=lambda item: item[0])
            anchors.append((int(round(cx)), int(round(cy))))
        else:
            anchors.append(((x0 + x1) // 2, (y0 + y1) // 2))
    return anchors


def draw_callout_pair(
    draw: ImageDraw.ImageDraw,
    top_boxes: list[tuple[int, int, int, int]],
    top_anchors: list[tuple[int, int]],
    bottom_boxes: list[tuple[int, int, int, int]],
) -> None:
    red = (210, 35, 35)
    for top_anchor, bottom_box in zip(top_anchors, bottom_boxes):
        draw.rectangle(bottom_box, outline=red, width=3)
        bottom_anchor = ((bottom_box[0] + bottom_box[2]) // 2, bottom_box[1])
        direction = np.array(bottom_anchor, dtype=float) - np.array(top_anchor, dtype=float)
        length = float(np.linalg.norm(direction))
        if length > 1e-6:
            direction /= length
            top_anchor = tuple((np.array(top_anchor, dtype=float) + direction * 22.0).round().astype(int))
        draw.line((top_anchor, bottom_anchor), fill=red, width=2)


def make_contact_sheet(
    top_paths: list[Path],
    section_paths: list[tuple[Path, Path]],
    output: Path,
) -> None:
    top_margin = 72
    col_w = 560
    top_h = 560
    gap = 8
    section_h = 225
    width = col_w * len(top_paths)
    height = top_margin + top_h + gap + section_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for i, top_path in enumerate(top_paths):
        x = i * col_w
        top_img, _ = fit_on_white_with_offset(top_path, (col_w, top_h))
        sheet.paste(top_img, (x, top_margin))
        top_boxes_local = colored_director_boxes(top_img)
        top_boxes = [
            (x + box[0], top_margin + box[1], x + box[2], top_margin + box[3])
            for box in top_boxes_local[:2]
        ]
        top_anchors = [
            (x + anchor[0], top_margin + anchor[1])
            for anchor in red_point_anchors(top_img, top_boxes_local[:2])
        ]

        bottom_y = top_margin + top_h + gap
        bottom_boxes = []
        for j, section_path in enumerate(section_paths[i]):
            box_x = x + j * (col_w // 2)
            section_img, _ = fit_on_white_with_offset(section_path, (col_w // 2, section_h))
            sheet.paste(section_img, (box_x, bottom_y))
            inset_pad = 7
            bottom_boxes.append(
                (
                    box_x + inset_pad,
                    bottom_y + inset_pad,
                    box_x + col_w // 2 - inset_pad,
                    bottom_y + section_h - inset_pad,
                )
            )
        draw_callout_pair(draw, top_boxes, top_anchors, bottom_boxes)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 46)
    except Exception:
        font = None
    axis_y = 70
    left_x = col_w // 2
    right_x = width - col_w // 2
    arrow = 20
    draw.line((left_x, axis_y, right_x - arrow, axis_y), fill=(30, 30, 30), width=5)
    draw.polygon(
        [(right_x, axis_y), (right_x - arrow, axis_y - arrow // 2), (right_x - arrow, axis_y + arrow // 2)],
        fill=(30, 30, 30),
    )
    label = "time"
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.text(((left_x + right_x - (bbox[2] - bbox[0])) / 2, 4), label, fill=(0, 0, 0), font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data_control_box")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data_control_box" / "loop_growth_local_clean",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=list(base.DEFAULT_STEPS))
    parser.add_argument("--target-center", type=float, nargs=3, default=base.DEFAULT_TARGET_CENTER.tolist())
    parser.add_argument("--half-width", type=float, default=13.0)
    parser.add_argument("--section-radius", type=float, default=2.4)
    parser.add_argument("--section-dr", type=float, default=0.8)
    parser.add_argument("--section-arc-dist", type=float, default=2.1)
    parser.add_argument("--section-rod-length", type=float, default=1.85)
    parser.add_argument("--section-rod-radius", type=float, default=0.075)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = np.asarray(args.target_center, dtype=float)
    infos: list[base.LoopInfo] = []
    q_snapshots: list[np.ndarray] = []

    for step in args.steps:
        q5 = np.load(args.data_dir / f"Q_{step}.npy")
        info = base.choose_loop(step, q5, target)
        target = info.center
        infos.append(info)
        q_snapshots.append(q5)

    box = base.make_render_box(infos, q_snapshots[0].shape[:3], float(args.half_width))
    bounds = base.local_bounds(box)
    view = base.camera_view_direction(base.CAMERA_AZIMUTH, base.CAMERA_ELEVATION)

    top_paths: list[Path] = []
    section_paths: list[tuple[Path, Path]] = []
    for q5, info in zip(q_snapshots, infos):
        top_path = args.output_dir / f"loop_callout_top_step_{info.step:05d}.png"
        base.render_loop(
            top_path,
            q5,
            info,
            box,
            half_width=float(args.half_width),
            director_spacing=2,
            show_axes=False,
            section_radius=float(args.section_radius),
            section_dr=float(args.section_dr),
            section_arc_dist=float(args.section_arc_dist),
            section_rod_length=float(args.section_rod_length),
            section_rod_radius=float(args.section_rod_radius),
        )

        q_obj = base.build_q_object(q5, info.step)
        line_index = base.choose_q_object_line_index(q_obj, info.center)
        line = q_obj.lines[line_index]
        u_pair = base.select_view_parallel_opposite_u_percents(line, view)
        section_a = args.output_dir / f"loop_callout_front_step_{info.step:05d}_a.png"
        section_b = args.output_dir / f"loop_callout_front_step_{info.step:05d}_b.png"
        front.render_front_section(
            section_a,
            q_obj,
            line_index,
            u_pair[0],
            bounds,
            step=info.step,
            section_radius=float(args.section_radius),
            section_dr=float(args.section_dr),
            section_arc_dist=float(args.section_arc_dist),
            rod_length=float(args.section_rod_length),
            rod_radius=float(args.section_rod_radius),
        )
        front.render_front_section(
            section_b,
            q_obj,
            line_index,
            u_pair[1],
            bounds,
            step=info.step,
            section_radius=float(args.section_radius),
            section_dr=float(args.section_dr),
            section_arc_dist=float(args.section_arc_dist),
            rod_length=float(args.section_rod_length),
            rod_radius=float(args.section_rod_radius),
        )

        top_paths.append(top_path)
        section_paths.append((section_a, section_b))
        print(f"step={info.step} top={top_path} u_pair=({u_pair[0]:.2f}, {u_pair[1]:.2f})")

    contact = args.output_dir / "loop_growth_3panel_cross_section_callouts.png"
    make_contact_sheet(top_paths, section_paths, contact)
    print(f"contact_sheet={contact}")


if __name__ == "__main__":
    main()
