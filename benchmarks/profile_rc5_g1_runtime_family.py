#!/usr/bin/env python3
"""Run one rc5 G-1 profile under an explicit TF32-off child policy."""

from __future__ import annotations

import torch

from benchmarks.profile_rc458_runtime_family import main as profile_main


def _state() -> tuple[str, bool, bool]:
    return (
        torch.get_float32_matmul_precision(),
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
    )


def main(argv: list[str] | None = None) -> int:
    before = _state()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if _state() != ("highest", False, False):
            raise RuntimeError("cannot enforce the frozen TF32-off profile policy")
        return profile_main(argv)
    finally:
        torch.set_float32_matmul_precision(before[0])
        torch.backends.cuda.matmul.allow_tf32 = before[1]
        torch.backends.cudnn.allow_tf32 = before[2]


if __name__ == "__main__":
    raise SystemExit(main())
