#!/usr/bin/env python3
"""Build a searchable verbatim PDF archive of the v0.2 architecture notes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cairo


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "PSSolver_v0_2_architecture_phase_archive_verbatim_zh.pdf"

SOURCE_ORDER = (
    "charter.md",
    "adr/0001-strangler-migration-over-v0-1-2.md",
    "adr/0002-problem-declarations-and-lowering.md",
    "adr/0003-plan-execution-state-and-workspace.md",
    "adr/0004-geometry-specific-capability-dispatch.md",
    "adr/0005-public-api-and-compatibility-policy.md",
    "adr/0006-identity-provenance-and-qualification.md",
    "adr/0007-run-spec-compatibility-facade.md",
    "adr/0008-tensor-free-system-declarations.md",
    "adr/0009-runtime-state-workspace-and-step-program.md",
    "adr/0010-sbdf2-and-modal-block-operators.md",
    "adr/0011-plane-compiled-v2-static-control.md",
    "api_surface_v0_1_2.md",
    "v0_1_2_oracle.md",
    "v0_1_2_oracle.json",
    "phase_1_transform_extraction.md",
    "phase_1_local_qualification.md",
    "phase_1_local_identity_control.json",
    "phase_1_h100_qualification.json",
    "phase_2_run_spec_decomposition.md",
    "phase_2_run_spec_inventory.json",
    "phase_2_qualification.md",
    "phase_2_local_qualification.md",
    "phase_2_local_qualification.json",
    "phase_2_h100_qualification.json",
    "phase_3_state_workspace_timestep.md",
    "phase_3_state_inventory.json",
    "phase_3_p34_h100_qualification.md",
    "phase_3_p34_h100_qualification.json",
    "phase_3_p35_h100_qualification.md",
    "phase_3_p35_h100_qualification.json",
    "phase_3_p36_local_qualification.md",
    "phase_3_p36_local_qualification.json",
    "phase_3_p36_performance_adjudication.md",
    "phase_3_p36_performance_adjudication.json",
    "phase_3_p36_closure_continuation.md",
    "phase_3_p36_closure_continuation.json",
    "phase_3_p36_reset_rebind_recovery.md",
    "phase_3_p36_reset_rebind_recovery.json",
    "phase_3_p36_reset_rebind_recovery_v3.md",
    "phase_3_p36_reset_rebind_recovery_v3.json",
    "phase_3_p36_reset_rebind_recovery_v4.md",
    "phase_3_p36_reset_rebind_recovery_v4.json",
    "phase_3_p36_reset_rebind_recovery_v5.md",
    "phase_3_p36_reset_rebind_recovery_v5.json",
    "phase_3_p36_reset_rebind_recovery_v6.md",
    "phase_3_p36_reset_rebind_recovery_v6.json",
    "phase_3_p36_final_closure.md",
    "phase_3_p36_final_closure.json",
    "phase_4_integrators_block_operators.md",
    "phase_4_contract.json",
    "phase_4_p41_integrator_history.md",
    "phase_4_p41_integrator_history.json",
    "phase_4_p42_scalar_sbdf2_reference.md",
    "phase_4_p42_scalar_sbdf2_reference.json",
    "phase_4_p43_convergence_restart.md",
    "phase_4_p43_convergence_restart.json",
    "phase_4_p44_modal_block_operator.md",
    "phase_4_p44_modal_block_operator.json",
    "phase_4_p44_h100_recovery.md",
    "phase_4_p44_h100_recovery.json",
    "phase_4_p44_contract_equivalent_adjudication.md",
    "phase_4_p44_contract_equivalent_adjudication.json",
    "phase_4_p45_combined_sbdf2_modal_block.md",
    "phase_4_p45_combined_sbdf2_modal_block.json",
    "phase_4_p46_closure_qualification.md",
    "phase_4_p46_closure_qualification.json",
    "phase_4_p46_memory_measurement_recovery.md",
    "phase_4_p46_memory_measurement_recovery.json",
    "phase_4_p46_final_closure.md",
    "phase_4_p46_final_closure.json",
    "phase_5_plane_compiled_v2.md",
    "phase_5_contract.json",
    "phase_5_plane_inventory.json",
    "phase_5_p51_compiled_declarations.md",
    "phase_5_p51_compiled_declarations.json",
    "phase_5_p52_construction_binding.md",
    "phase_5_p52_construction_binding.json",
    "phase_5_p53_compiled_euler_step.md",
    "phase_5_p53_compiled_euler_step.json",
    "phase_5_p54_observation_checkpoint.md",
    "phase_5_p54_observation_checkpoint.json",
    "phase_5_p55_application_connection.md",
    "phase_5_p55_application_connection.json",
    "phase_5_p56_local_closure.md",
    "phase_5_p56_local_closure.json",
    "phase_6_plane_compiled_v2_qualification_plan.md",
    "phase_6_plane_compiled_v2_qualification_plan.json",
    "phase_6_p61_h100_qualification_support.md",
    "phase_6_p61_h100_qualification_support.json",
    "phase_6_p61_performance_equivalence_recovery.md",
    "phase_6_p61_performance_equivalence_recovery.json",
    "phase_6_p61_h100_qualification_result.md",
    "phase_6_p61_h100_qualification_result.json",
    "phase_6_p62_long_run_equivalence.md",
    "phase_6_p62_long_run_equivalence.json",
    "phase_6_final_closure.md",
    "phase_6_final_closure.json",
    "phase_7_channel_inventory.json",
    "phase_7_channel_migration_plan.md",
    "phase_7_channel_migration_plan.json",
    "phase_7_p71_run_spec_decomposition.md",
    "phase_7_p71_run_spec_decomposition.json",
    "phase_7_p72_channel_stokes_extraction.md",
    "phase_7_p72_channel_stokes_extraction.json",
    "phase_7_p73_channel_compiled_execution.md",
    "phase_7_p73_channel_compiled_execution.json",
    "phase_7_p74_channel_runtime_facade.md",
    "phase_7_p74_channel_runtime_facade.json",
    "phase_7_p75_local_closure.md",
    "phase_7_p75_local_closure.json",
    "phase_7_p76_h100_qualification_plan.md",
    "phase_7_p76_h100_qualification_plan.json",
    "phase_7_p76_transform_call_recovery.md",
    "phase_7_p76_transform_call_recovery.json",
)

PAGE_WIDTH = 595.276
PAGE_HEIGHT = 841.890
MARGIN_LEFT = 42.0
MARGIN_RIGHT = 42.0
MARGIN_TOP = 48.0
MARGIN_BOTTOM = 42.0
BODY_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
BODY_BOTTOM = PAGE_HEIGHT - MARGIN_BOTTOM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArchiveRenderer:
    def __init__(self, output: Path) -> None:
        self.surface = cairo.PDFSurface(str(output), PAGE_WIDTH, PAGE_HEIGHT)
        self.context = cairo.Context(self.surface)
        self.page_number = 0
        self.section = ""
        self.y = MARGIN_TOP
        self.body_font = ("Noto Sans Mono CJK SC", 7.4, cairo.FONT_WEIGHT_NORMAL)
        self.small_font = ("Noto Sans CJK SC", 7.0, cairo.FONT_WEIGHT_NORMAL)
        self.header_font = ("Noto Sans CJK SC", 8.0, cairo.FONT_WEIGHT_NORMAL)
        self.title_font = ("Noto Sans CJK SC", 20.0, cairo.FONT_WEIGHT_BOLD)
        self.section_font = ("Noto Sans CJK SC", 13.0, cairo.FONT_WEIGHT_BOLD)

    def _select_font(self, descriptor: tuple[str, float, int]) -> None:
        family, size, weight = descriptor
        self.context.select_font_face(
            family,
            cairo.FONT_SLANT_NORMAL,
            weight,
        )
        self.context.set_font_size(size)

    def _text_width(self, text: str, descriptor: tuple[str, float, int]) -> float:
        self._select_font(descriptor)
        return float(self.context.text_extents(text).x_advance)

    def _wrapped_fragments(
        self,
        text: str,
        descriptor: tuple[str, float, int],
        *,
        width: float = BODY_WIDTH,
    ) -> list[str]:
        if not text:
            return [" "]
        fragments: list[str] = []
        remaining = text
        while remaining:
            if self._text_width(remaining, descriptor) <= width:
                fragments.append(remaining)
                break
            low, high = 1, len(remaining)
            while low < high:
                middle = (low + high + 1) // 2
                if self._text_width(remaining[:middle], descriptor) <= width:
                    low = middle
                else:
                    high = middle - 1
            split = max(1, low)
            fragments.append(remaining[:split])
            remaining = remaining[split:]
        return fragments

    def _draw_text(
        self,
        text: str,
        descriptor: tuple[str, float, int],
        x: float,
        baseline: float,
    ) -> None:
        self._select_font(descriptor)
        self.context.move_to(x, baseline)
        self.context.show_text(text)

    def _footer(self) -> None:
        footer = f"PSSolver v0.2 architecture archive  |  page {self.page_number}"
        self.context.set_source_rgb(0.35, 0.35, 0.35)
        self._draw_text(
            footer,
            self.small_font,
            PAGE_WIDTH - MARGIN_RIGHT - self._text_width(footer, self.small_font),
            PAGE_HEIGHT - 22,
        )

    def _header(self) -> None:
        if not self.section:
            return
        self.context.set_source_rgb(0.3, 0.3, 0.3)
        fragments = self._wrapped_fragments(
            self.section,
            self.header_font,
            width=BODY_WIDTH,
        )
        self._draw_text(fragments[0], self.header_font, MARGIN_LEFT, 24)

    def new_page(self, *, section: str = "") -> None:
        if self.page_number:
            self._footer()
            self.surface.show_page()
        self.page_number += 1
        self.section = section
        self.context.set_source_rgb(1, 1, 1)
        self.context.paint()
        self.context.set_source_rgb(0, 0, 0)
        self._header()
        self.y = MARGIN_TOP

    def ensure_space(self, required: float, *, section: str | None = None) -> None:
        if self.y + required <= BODY_BOTTOM:
            return
        self.new_page(section=self.section if section is None else section)

    def paragraph(
        self,
        text: str,
        font_description: tuple[str, float, int],
        *,
        gap_after: float = 7.0,
    ) -> None:
        fragments = self._wrapped_fragments(text, font_description)
        line_height = font_description[1] * 1.45
        for fragment in fragments:
            self.ensure_space(line_height)
            self.context.set_source_rgb(0, 0, 0)
            self._draw_text(
                fragment,
                font_description,
                MARGIN_LEFT,
                self.y + font_description[1],
            )
            self.y += line_height
        self.y += gap_after

    def source_line(self, text: str) -> None:
        fragments = self._wrapped_fragments(text, self.body_font)
        line_height = self.body_font[1] * 1.42
        for fragment in fragments:
            self.ensure_space(line_height)
            self.context.set_source_rgb(0.08, 0.08, 0.08)
            self._draw_text(
                fragment,
                self.body_font,
                MARGIN_LEFT,
                self.y + self.body_font[1],
            )
            self.y += line_height

    def close(self) -> None:
        self._footer()
        self.surface.finish()


def validated_sources() -> list[Path]:
    sources = [ROOT / relative for relative in SOURCE_ORDER]
    missing = [str(path.relative_to(ROOT)) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing archive sources: {missing!r}")
    discovered = {
        path.relative_to(ROOT).as_posix()
        for pattern in ("*.md", "*.json")
        for path in ROOT.rglob(pattern)
        if path.name != Path(__file__).name
    }
    expected = set(SOURCE_ORDER)
    unexpected = sorted(discovered - expected)
    if unexpected:
        raise RuntimeError(
            "SOURCE_ORDER must explicitly classify every Markdown/JSON source; "
            f"unlisted files: {unexpected!r}"
        )
    return sources


def main() -> None:
    sources = validated_sources()
    renderer = ArchiveRenderer(OUTPUT)

    renderer.new_page(section="Archive cover")
    renderer.y = 120
    renderer.paragraph(
        "PSSolver v0.2 架构与 Phase 记录逐字归档",
        renderer.title_font,
        gap_after=18,
    )
    renderer.paragraph(
        "Verbatim archive of notes/architecture_v0_2",
        ("Noto Sans CJK SC", 12.0, cairo.FONT_WEIGHT_NORMAL),
        gap_after=22,
    )
    renderer.paragraph(
        "正文按 UTF-8 源文件逐行排版；Markdown 标记、代码围栏、JSON 标点与原文字符均不改写。视觉自动换行不代表源文件新增换行。",
        ("Noto Sans CJK SC", 9.0, cairo.FONT_WEIGHT_NORMAL),
        gap_after=14,
    )
    renderer.paragraph(
        f"Source files: {len(sources)}",
        ("Noto Sans CJK SC", 9.0, cairo.FONT_WEIGHT_NORMAL),
    )

    renderer.new_page(section="Source manifest")
    renderer.paragraph("来源清单与 SHA-256", renderer.section_font, gap_after=12)
    for index, path in enumerate(sources, start=1):
        relative = path.relative_to(ROOT).as_posix()
        renderer.source_line(f"{index:02d}  {relative}")
        renderer.source_line(f"    sha256  {sha256(path)}")
        renderer.source_line("")

    renderer.new_page(section="Contents")
    renderer.paragraph("文件顺序", renderer.section_font, gap_after=12)
    for index, path in enumerate(sources, start=1):
        renderer.source_line(f"{index:02d}  {path.relative_to(ROOT).as_posix()}")

    for index, path in enumerate(sources, start=1):
        relative = path.relative_to(ROOT).as_posix()
        renderer.new_page(section=relative)
        renderer.paragraph(
            f"SOURCE {index:02d}/{len(sources):02d}: {relative}",
            renderer.section_font,
            gap_after=12,
        )
        text = path.read_text(encoding="utf-8")
        for line in text.split("\n"):
            renderer.source_line(line)

    renderer.close()
    print(OUTPUT)
    print(f"pages={renderer.page_number}")
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
