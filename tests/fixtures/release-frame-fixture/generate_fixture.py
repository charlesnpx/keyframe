#!/usr/bin/env python3
"""Render the public synthetic release-frame fixture and encode its MP4."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1280
HEIGHT = 720
FPS = 30
BACKGROUND = "#f2f5f9"
INK = "#18212f"
MUTED = "#647184"
PANEL = "#ffffff"
BORDER = "#c9d2df"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    raise RuntimeError(
        "fixture generation requires Arial or DejaVu Sans; no supported font was found"
    )


def _rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    radius: int,
    fill: str,
    outline: str | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def _render_state(state: dict[str, object], sequence: int, output: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    heading_font = _font(52, bold=True)
    eyebrow_font = _font(22, bold=True)
    label_font = _font(28, bold=True)
    value_font = _font(30)
    footer_font = _font(20)

    accent = str(state["accent"])
    draw.rectangle((0, 0, WIDTH, 14), fill=accent)
    draw.text(
        (72, 48),
        str(state["eyebrow"]),
        font=eyebrow_font,
        fill=accent,
    )
    draw.text(
        (72, 88),
        str(state["heading"]),
        font=heading_font,
        fill=INK,
    )
    draw.text(
        (1120, 58),
        f"0{sequence} / 06",
        font=eyebrow_font,
        fill=MUTED,
    )

    panel = (64, 178, 1216, 646)
    _rounded_rectangle(
        draw,
        panel,
        radius=22,
        fill=PANEL,
        outline=BORDER,
        width=2,
    )

    fields = list(state["fields"])
    row_height = 105
    for index, raw_field in enumerate(fields):
        label, value = (str(raw_field[0]), str(raw_field[1]))
        top = 198 + index * row_height
        if index:
            draw.line((92, top - 10, 1188, top - 10), fill=BORDER, width=2)
        draw.text((96, top + 18), f"{label}:", font=label_font, fill=INK)
        value_box = (650, top + 4, 1178, top + 80)
        _rounded_rectangle(
            draw,
            value_box,
            radius=12,
            fill="#f8fafc",
            outline=accent if value else BORDER,
            width=3 if value else 2,
        )
        if value:
            draw.text((674, top + 23), value, font=value_font, fill=INK)
        else:
            draw.line(
                (680, top + 58, 1148, top + 58),
                fill="#aeb9c7",
                width=3,
            )

    draw.text(
        (72, 674),
        "Synthetic public fixture • no customer or production data",
        font=footer_font,
        fill=MUTED,
    )
    image.save(output, format="PNG", optimize=False)


def _render_transition(output: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#172235")
    draw = ImageDraw.Draw(image)
    heading_font = _font(56, bold=True)
    body_font = _font(28)
    draw.rectangle((0, 0, WIDTH, 18), fill="#5b8def")
    draw.text(
        (256, 282),
        "KEYFRAME RELEASE FIXTURE",
        font=heading_font,
        fill="#ffffff",
    )
    draw.text(
        (430, 370),
        "FORM TRANSITION",
        font=body_font,
        fill="#c8d7ef",
    )
    image.save(output, format="PNG", optimize=False)


def _encode(states_dir: Path, output: Path, ffmpeg: str) -> None:
    transition = states_dir / "transition.png"
    timeline = states_dir / "timeline.txt"
    lines: list[str] = []
    for sequence in range(1, 6):
        lines.extend(
            [
                f"file '{states_dir / f'state-{sequence:02d}.png'}'",
                "duration 6",
            ]
        )
    lines.extend(
        [
            f"file '{transition}'",
            "duration 0.75",
            f"file '{states_dir / 'state-06.png'}'",
            "duration 4.5",
            f"file '{transition}'",
            "duration 0.75",
            f"file '{transition}'",
        ]
    )
    timeline.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(timeline),
        "-t",
        "36",
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-preset",
        "veryslow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("source.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("release-frame-fixture.mp4"),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)

    source = json.loads(args.source.read_text(encoding="utf-8"))
    states = source.get("states")
    if not isinstance(states, list) or len(states) != 6:
        raise ValueError("fixture source must define exactly six states")
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise RuntimeError(f"ffmpeg executable not found: {args.ffmpeg}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="keyframe-release-fixture-") as raw:
        states_dir = Path(raw)
        for sequence, state in enumerate(states, start=1):
            if not isinstance(state, dict):
                raise ValueError(f"fixture state {sequence} must be an object")
            _render_state(
                state,
                sequence,
                states_dir / f"state-{sequence:02d}.png",
            )
        _render_transition(states_dir / "transition.png")
        _encode(states_dir, args.output, ffmpeg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
