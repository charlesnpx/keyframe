"""Standalone fresh and replay validation for the public frame fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
import unicodedata
import urllib.parse
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


FIXTURE_IDENTIFIER = "keyframe.release-frame-fixture"
FIXTURE_SCHEMA_VERSION = 1
EVIDENCE_IDENTIFIER = "keyframe.frame-fixture-evidence"
EVIDENCE_SCHEMA_VERSION = 1
SUPPORTED_PLATFORMS = {
    ("Darwin", "arm64"): "apple-vision",
    ("Linux", "x86_64"): "paddleocr",
}
REQUIRED_TARGETS = (
    ("source-fields", 3.0),
    ("priority-sections", 9.0),
    ("consequence-comments", 15.0),
    ("page-transition", 21.0),
    ("signed-blank", 27.0),
    ("signed-populated", 33.0),
)
DEFAULT_CONFIGURATION = {
    "routing": "default",
    "entrypoint": "python -m keyframe.cli",
    "frames_only": True,
    "sample_interval_seconds": 0.5,
    "pass1_clusters": 15,
    "max_output_frames": 15,
    "verbose_trace": True,
}
RUNTIME_PACKAGE_NAMES = {
    "keyframe": "keyframe",
    "torch": "torch",
    "transformers": "transformers",
    "open_clip_torch": "open-clip-torch",
    "pillow": "Pillow",
    "opencv_python": "opencv-python",
    "paddleocr": "paddleocr",
    "paddlepaddle": "paddlepaddle",
    "pyobjc_framework_vision": "pyobjc-framework-Vision",
    "pyobjc_framework_quartz": "pyobjc-framework-Quartz",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class ReleaseEvidenceError(RuntimeError):
    """The release-frame evidence contract could not be evaluated."""


@dataclass(frozen=True)
class FixtureContract:
    root: Path
    metadata_path: Path
    metadata: dict[str, Any]
    recording_path: Path
    license_path: Path
    source_paths: tuple[Path, ...]
    media_probe: dict[str, Any]


@dataclass(frozen=True)
class ReplayResult:
    report: dict[str, Any] | None
    recomputed: dict[str, Any] | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def validation(self) -> dict[str, Any]:
        return {"passed": self.passed, "failures": list(self.failures)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseEvidenceError(f"{label} is not valid UTF-8 JSON: {exc}") from exc


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseEvidenceError(f"{label} must be an object")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise ReleaseEvidenceError(f"{label} must be a finite number")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ReleaseEvidenceError(f"{label} must be a finite number")
    return rendered


def _true_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReleaseEvidenceError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _prepare_empty_directory(path: Path, label: str) -> Path:
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_dir():
            raise ReleaseEvidenceError(f"{label} must be a real directory: {path}")
        if any(path.iterdir()):
            raise ReleaseEvidenceError(f"{label} must be empty: {path}")
    else:
        path.mkdir(parents=True)
    return path.resolve()


def _relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ReleaseEvidenceError(f"{label} must be a nonempty relative path")
    if "\\" in value:
        raise ReleaseEvidenceError(f"{label} must use portable '/' separators")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ReleaseEvidenceError(f"{label} must remain beneath its bundle")
    if any(":" in part for part in relative.parts):
        raise ReleaseEvidenceError(f"{label} must remain beneath its bundle")
    return relative


def _resolve_bundle_file(bundle: Path, value: Any, label: str) -> Path:
    bundle = Path(bundle)
    if bundle.is_symlink():
        raise ReleaseEvidenceError(f"{label} bundle root must not be a symlink")
    try:
        bundle_mode = bundle.stat().st_mode
    except OSError as exc:
        raise ReleaseEvidenceError(f"{label} bundle root is unavailable: {exc}") from exc
    if not stat.S_ISDIR(bundle_mode):
        raise ReleaseEvidenceError(f"{label} bundle root must be a directory")

    relative = _relative_path(value, label)
    current = bundle
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ReleaseEvidenceError(f"{label} is unavailable: {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ReleaseEvidenceError(f"{label} must not traverse a symlink: {current}")
        final = index == len(relative.parts) - 1
        if final:
            if not stat.S_ISREG(mode):
                raise ReleaseEvidenceError(f"{label} must be a regular file: {current}")
        elif not stat.S_ISDIR(mode):
            raise ReleaseEvidenceError(
                f"{label} parent must be a real directory: {current}"
            )

    resolved_bundle = bundle.resolve(strict=True)
    resolved_file = current.resolve(strict=True)
    if not resolved_file.is_relative_to(resolved_bundle):
        raise ReleaseEvidenceError(f"{label} resolves outside its bundle")
    return resolved_file


def resolve_bundle_file(
    bundle: str | Path,
    relative_path: Any,
    *,
    label: str = "bundle artifact",
) -> Path:
    """Resolve one regular, non-symlinked file beneath a bundle root."""

    return _resolve_bundle_file(Path(bundle), relative_path, label)


def _portable_relative(path: Path, root: Path) -> str:
    return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()


def normalize_tokens(value: str) -> tuple[str, ...]:
    """Normalize OCR text across Apple Vision and PaddleOCR tokenization."""

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    characters = [
        character if character.isalnum() else " "
        for character in normalized
    ]
    return tuple("".join(characters).split())


def _contains_variant(haystack: tuple[str, ...], variant: str) -> bool:
    needle = normalize_tokens(variant)
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        haystack[index : index + width] == needle
        for index in range(len(haystack) - width + 1)
    )


def token_matches(
    haystack: tuple[str, ...],
    canonical: str,
    aliases: Mapping[str, Sequence[str]],
) -> bool:
    variants = (canonical, *aliases.get(canonical, ()))
    return any(_contains_variant(haystack, variant) for variant in variants)


def _validate_aliases(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise ReleaseEvidenceError("fixture aliases must be an object")
    if len(value) > 16:
        raise ReleaseEvidenceError("fixture aliases may define at most 16 tokens")
    normalized: dict[str, tuple[str, ...]] = {}
    for raw_token, raw_aliases in value.items():
        if (
            not isinstance(raw_token, str)
            or normalize_tokens(raw_token) != (raw_token,)
        ):
            raise ReleaseEvidenceError(
                "fixture alias keys must be normalized single tokens"
            )
        if not isinstance(raw_aliases, list) or len(raw_aliases) > 4:
            raise ReleaseEvidenceError(
                f"fixture alias {raw_token!r} may contain at most four strings"
            )
        aliases: list[str] = []
        for alias in raw_aliases:
            if not isinstance(alias, str) or not normalize_tokens(alias):
                raise ReleaseEvidenceError(
                    f"fixture alias {raw_token!r} contains an invalid variant"
                )
            if alias in aliases:
                raise ReleaseEvidenceError(
                    f"fixture alias {raw_token!r} contains a duplicate variant"
                )
            aliases.append(alias)
        normalized[raw_token] = tuple(aliases)
    return normalized


def _validate_targets(value: Any, aliases: Mapping[str, Sequence[str]]) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or len(value) != len(REQUIRED_TARGETS):
        raise ReleaseEvidenceError("fixture metadata must define exactly six targets")
    normalized: list[dict[str, Any]] = []
    for index, (raw, required) in enumerate(zip(value, REQUIRED_TARGETS, strict=True)):
        target = _mapping(raw, f"fixture target {index}")
        target_id, center = required
        if target.get("id") != target_id:
            raise ReleaseEvidenceError(
                f"fixture target {index} id must be {target_id!r}"
            )
        time_seconds = _finite_number(
            target.get("time_seconds"),
            f"fixture target {target_id} time_seconds",
        )
        tolerance = _finite_number(
            target.get("tolerance_seconds"),
            f"fixture target {target_id} tolerance_seconds",
        )
        if time_seconds != center:
            raise ReleaseEvidenceError(
                f"fixture target {target_id} must be centered at {center:g} seconds"
            )
        if tolerance != 2.25:
            raise ReleaseEvidenceError(
                f"fixture target {target_id} tolerance must be 2.25 seconds"
            )
        description = target.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ReleaseEvidenceError(
                f"fixture target {target_id} description must be nonempty"
            )
        raw_groups = target.get("required_token_groups")
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ReleaseEvidenceError(
                f"fixture target {target_id} must define token groups"
            )
        groups: list[dict[str, Any]] = []
        group_names: set[str] = set()
        for group_index, raw_group in enumerate(raw_groups):
            group = _mapping(
                raw_group,
                f"fixture target {target_id} token group {group_index}",
            )
            name = group.get("name")
            tokens = group.get("tokens")
            if not isinstance(name, str) or not name.strip() or name in group_names:
                raise ReleaseEvidenceError(
                    f"fixture target {target_id} token group names must be unique"
                )
            if not isinstance(tokens, list) or not tokens:
                raise ReleaseEvidenceError(
                    f"fixture target {target_id} token group {name!r} is empty"
                )
            if any(
                not isinstance(token, str) or normalize_tokens(token) != (token,)
                for token in tokens
            ):
                raise ReleaseEvidenceError(
                    f"fixture target {target_id} token group {name!r} "
                    "must contain normalized single tokens"
                )
            unknown_aliases = set(tokens) & set(aliases)
            del unknown_aliases  # Aliases are optional for any canonical token.
            group_names.add(name)
            groups.append({"name": name, "tokens": list(tokens)})
        normalized.append(
            {
                "id": target_id,
                "time_seconds": time_seconds,
                "tolerance_seconds": tolerance,
                "description": description,
                "required_token_groups": groups,
            }
        )
    return tuple(normalized)


def _validate_budgets(value: Any) -> dict[str, Any]:
    budgets = _mapping(value, "fixture budgets")
    minimum = _true_integer(
        budgets.get("min_output_frames"),
        "fixture budgets.min_output_frames",
        minimum=1,
    )
    maximum = _true_integer(
        budgets.get("max_output_frames"),
        "fixture budgets.max_output_frames",
        minimum=1,
    )
    redundancy = _finite_number(
        budgets.get("max_target_redundancy"),
        "fixture budgets.max_target_redundancy",
    )
    if minimum > maximum:
        raise ReleaseEvidenceError(
            "fixture minimum output budget cannot exceed its maximum"
        )
    if not 0 <= redundancy <= 1:
        raise ReleaseEvidenceError(
            "fixture max target redundancy must be within [0, 1]"
        )
    return {
        "min_output_frames": minimum,
        "max_output_frames": maximum,
        "max_target_redundancy": redundancy,
    }


def _ffprobe(path: Path) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise ReleaseEvidenceError("ffprobe is required to validate the fixture")
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ReleaseEvidenceError(f"ffprobe could not read fixture media: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceError("ffprobe returned invalid JSON") from exc
    return _mapping(payload, "ffprobe result")


def _media_identity(probe: Mapping[str, Any]) -> dict[str, Any]:
    streams = probe.get("streams")
    format_record = probe.get("format")
    if not isinstance(streams, list) or not isinstance(format_record, dict):
        raise ReleaseEvidenceError("ffprobe result is missing streams or format")
    videos = [
        stream for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    audios = [
        stream for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "audio"
    ]
    if len(videos) != 1:
        raise ReleaseEvidenceError("fixture must contain exactly one video stream")
    video = videos[0]
    duration = video.get("duration", format_record.get("duration"))
    frame_count = video.get("nb_frames")
    try:
        parsed_duration = float(duration)
        parsed_frames = int(frame_count)
    except (TypeError, ValueError) as exc:
        raise ReleaseEvidenceError(
            "fixture media must expose duration and frame count"
        ) from exc
    return {
        "container": "mp4"
        if "mp4" in str(format_record.get("format_name", "")).split(",")
        else str(format_record.get("format_name", "")),
        "codec": video.get("codec_name"),
        "pixel_format": video.get("pix_fmt"),
        "width": video.get("width"),
        "height": video.get("height"),
        "duration_seconds": parsed_duration,
        "frame_rate": video.get("avg_frame_rate"),
        "frame_count": parsed_frames,
        "video_streams": len(videos),
        "audio_streams": len(audios),
    }


def load_fixture_contract(
    metadata_path: str | Path,
    *,
    recording_path: str | Path | None = None,
) -> FixtureContract:
    """Validate fixture metadata, provenance, hashes, and media identity."""

    metadata_path = Path(metadata_path).expanduser()
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ReleaseEvidenceError(
            f"fixture metadata must be a regular non-symlinked file: {metadata_path}"
        )
    metadata_path = metadata_path.resolve()
    root = metadata_path.parent
    metadata = _mapping(_load_json(metadata_path, "fixture metadata"), "fixture metadata")
    if metadata.get("identifier") != FIXTURE_IDENTIFIER:
        raise ReleaseEvidenceError(
            f"fixture identifier must be {FIXTURE_IDENTIFIER!r}"
        )
    if metadata.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ReleaseEvidenceError(
            f"fixture schema_version must be {FIXTURE_SCHEMA_VERSION}"
        )
    creator = metadata.get("creator")
    if not isinstance(creator, str) or not creator.strip():
        raise ReleaseEvidenceError("fixture creator must be nonempty")

    license_record = _mapping(metadata.get("license"), "fixture license")
    if license_record.get("spdx") != "MIT":
        raise ReleaseEvidenceError("fixture redistribution license must be MIT")
    if license_record.get("redistribution_permitted") is not True:
        raise ReleaseEvidenceError(
            "fixture metadata must explicitly permit redistribution"
        )
    license_path = _resolve_bundle_file(
        root,
        license_record.get("path"),
        "fixture license",
    )
    if license_record.get("sha256") != _sha256(license_path):
        raise ReleaseEvidenceError("fixture license hash does not match metadata")

    provenance = _mapping(metadata.get("provenance"), "fixture provenance")
    for field in ("ownership", "construction_command"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise ReleaseEvidenceError(f"fixture provenance.{field} must be nonempty")
    raw_sources = provenance.get("source_files")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ReleaseEvidenceError("fixture provenance must name source files")
    source_paths: list[Path] = []
    for index, raw_source in enumerate(raw_sources):
        source = _mapping(raw_source, f"fixture source {index}")
        source_path = _resolve_bundle_file(
            root,
            source.get("path"),
            f"fixture source {index}",
        )
        if source.get("sha256") != _sha256(source_path):
            raise ReleaseEvidenceError(
                f"fixture source {source_path.name} hash does not match metadata"
            )
        source_paths.append(source_path)

    recording = _mapping(metadata.get("recording"), "fixture recording")
    declared_recording = _resolve_bundle_file(
        root,
        recording.get("path"),
        "fixture recording",
    )
    if recording_path is not None:
        supplied = Path(recording_path).expanduser()
        if supplied.is_symlink() or not supplied.is_file():
            raise ReleaseEvidenceError(
                f"fixture recording must be a regular non-symlinked file: {supplied}"
            )
        declared_recording = supplied.resolve()
    expected_size = _true_integer(
        recording.get("size_bytes"),
        "fixture recording.size_bytes",
        minimum=1,
    )
    expected_hash = recording.get("sha256")
    if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
        raise ReleaseEvidenceError("fixture recording.sha256 must be lowercase SHA-256")
    if declared_recording.stat().st_size != expected_size:
        raise ReleaseEvidenceError("fixture recording size does not match metadata")
    if _sha256(declared_recording) != expected_hash:
        raise ReleaseEvidenceError("fixture recording hash does not match metadata")

    aliases = _validate_aliases(metadata.get("aliases"))
    _validate_targets(metadata.get("targets"), aliases)
    _validate_budgets(metadata.get("budgets"))
    declared_media = _mapping(metadata.get("media"), "fixture media")
    probe = _ffprobe(declared_recording)
    observed_media = _media_identity(probe)
    expected_media = {
        "container": "mp4",
        "codec": "h264",
        "pixel_format": "yuv420p",
        "width": 1280,
        "height": 720,
        "duration_seconds": 36.0,
        "frame_rate": "30/1",
        "frame_count": 1080,
        "video_streams": 1,
        "audio_streams": 0,
    }
    if set(declared_media) != set(expected_media):
        raise ReleaseEvidenceError("fixture media metadata has unexpected fields")
    for field, expected in expected_media.items():
        declared = declared_media.get(field)
        observed = observed_media.get(field)
        if field == "duration_seconds":
            if (
                not math.isclose(
                    _finite_number(declared, "fixture media.duration_seconds"),
                    expected,
                    rel_tol=0.0,
                    abs_tol=0.01,
                )
                or not math.isclose(
                    _finite_number(observed, "observed fixture duration"),
                    expected,
                    rel_tol=0.0,
                    abs_tol=0.01,
                )
            ):
                raise ReleaseEvidenceError("fixture duration must be exactly 36 seconds")
        elif declared != expected or observed != expected:
            raise ReleaseEvidenceError(
                f"fixture media {field} must be {expected!r}; "
                f"declared {declared!r}, observed {observed!r}"
            )

    return FixtureContract(
        root=root,
        metadata_path=metadata_path,
        metadata=metadata,
        recording_path=declared_recording,
        license_path=license_path,
        source_paths=tuple(source_paths),
        media_probe=probe,
    )


def qa_targets_from_fixture(metadata: Mapping[str, Any]) -> dict[str, Any]:
    aliases = _validate_aliases(metadata.get("aliases"))
    targets = _validate_targets(metadata.get("targets"), aliases)
    return {
        "targets": [
            {
                "time": target["time_seconds"],
                "label": target["id"],
                "tolerance": target["tolerance_seconds"],
                "anchor_tokens": [
                    token
                    for group in target["required_token_groups"]
                    for token in group["tokens"]
                ],
            }
            for target in targets
        ]
    }


def _png_dimensions(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as handle:
            signature = handle.read(8)
            length = int.from_bytes(handle.read(4), "big")
            chunk_type = handle.read(4)
            dimensions = handle.read(8)
    except OSError as exc:
        raise ReleaseEvidenceError(f"could not read PNG artifact {path}: {exc}") from exc
    if (
        signature != b"\x89PNG\r\n\x1a\n"
        or chunk_type != b"IHDR"
        or length != 13
        or len(dimensions) != 8
    ):
        raise ReleaseEvidenceError(f"frame artifact is not a valid PNG: {path}")
    return (
        int.from_bytes(dimensions[:4], "big"),
        int.from_bytes(dimensions[4:], "big"),
    )


def _artifact_file_record(path: Path, bundle: Path) -> dict[str, Any]:
    return {
        "path": _portable_relative(path, bundle),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _fixture_record(contract: FixtureContract, bundle: Path) -> dict[str, Any]:
    supporting = [contract.license_path, *contract.source_paths]
    return {
        "metadata": _artifact_file_record(contract.metadata_path, bundle),
        "recording": _artifact_file_record(contract.recording_path, bundle),
        "supporting_files": [
            _artifact_file_record(path, bundle)
            for path in supporting
        ],
        "media": _media_identity(contract.media_probe),
    }


def _frame_rows(
    manifest: Mapping[str, Any],
    captions: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("frames"), list
    ):
        raise ReleaseEvidenceError("frame manifest must be a schema_version 1 object")
    if not isinstance(captions, list):
        raise ReleaseEvidenceError("captions artifact must be a list")
    manifest_rows: list[dict[str, Any]] = []
    manifest_names: set[str] = set()
    for index, raw in enumerate(manifest["frames"]):
        row = _mapping(raw, f"manifest frame {index}")
        name = row.get("filename")
        if not isinstance(name, str) or PurePosixPath(name).name != name:
            raise ReleaseEvidenceError(
                f"manifest frame {index} filename must be a basename"
            )
        if name in manifest_names:
            raise ReleaseEvidenceError(f"manifest contains duplicate frame {name!r}")
        timestamp = _finite_number(
            row.get("timestamp"),
            f"manifest frame {name} timestamp",
        )
        raw_merged_timestamps = row.get("merged_timestamps", [timestamp])
        if not isinstance(raw_merged_timestamps, list):
            raise ReleaseEvidenceError(
                f"manifest frame {name} merged_timestamps must be a list"
            )
        merged_timestamps = [
            _finite_number(
                value,
                f"manifest frame {name} merged timestamp",
            )
            for value in raw_merged_timestamps
        ]
        if any(value < 0 or value > 36.0 for value in merged_timestamps):
            raise ReleaseEvidenceError(
                f"manifest frame {name} merged timestamps must be within the fixture"
            )
        if not any(
            math.isclose(
                value,
                timestamp,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for value in merged_timestamps
        ):
            raise ReleaseEvidenceError(
                f"manifest frame {name} merged timestamps must include its timestamp"
            )
        manifest_names.add(name)
        manifest_rows.append(
            {
                **row,
                "filename": name,
                "timestamp": timestamp,
                "merged_timestamps": merged_timestamps,
            }
        )

    caption_rows: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(captions):
        row = _mapping(raw, f"caption frame {index}")
        name = row.get("file")
        if not isinstance(name, str) or PurePosixPath(name).name != name:
            raise ReleaseEvidenceError(
                f"caption frame {index} file must be a basename"
            )
        if name in caption_rows:
            raise ReleaseEvidenceError(f"captions contain duplicate frame {name!r}")
        timestamp = _finite_number(
            row.get("timestamp"),
            f"caption frame {name} timestamp",
        )
        caption_rows[name] = {**row, "file": name, "timestamp": timestamp}
    if manifest_names != set(caption_rows):
        raise ReleaseEvidenceError(
            "manifest and captions must index the same PNG file set"
        )
    for row in manifest_rows:
        other = caption_rows[row["filename"]]
        if not math.isclose(
            row["timestamp"],
            other["timestamp"],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ReleaseEvidenceError(
                f"manifest and captions disagree on {row['filename']} timestamp"
            )
    return manifest_rows, caption_rows


def _text_for_frame(
    manifest_row: Mapping[str, Any],
    caption_row: Mapping[str, Any],
) -> str:
    values: list[str] = []
    for row in (manifest_row, caption_row):
        for field in ("ocr_text", "caption"):
            value = row.get(field)
            if isinstance(value, str):
                values.append(value)
        raw_tokens = row.get("ocr_tokens")
        if isinstance(raw_tokens, list):
            values.extend(str(token) for token in raw_tokens if isinstance(token, str))
    return "\n".join(values)


def _evaluate_targets(
    metadata: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    caption_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    aliases = _validate_aliases(metadata.get("aliases"))
    targets = _validate_targets(metadata.get("targets"), aliases)
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for target in targets:
        candidate_results: list[
            tuple[
                int,
                float,
                str,
                float,
                float,
                str,
                list[dict[str, Any]],
            ]
        ] = []
        for row in manifest_rows:
            selected_timestamp = float(row["timestamp"])
            lineage_timestamps = [
                float(value)
                for value in row.get("merged_timestamps", [selected_timestamp])
            ]
            matched_timestamp = min(
                lineage_timestamps,
                key=lambda value: (
                    abs(value - float(target["time_seconds"])),
                    value,
                ),
            )
            delta = abs(
                matched_timestamp - float(target["time_seconds"])
            )
            if delta > float(target["tolerance_seconds"]):
                continue
            tokens = normalize_tokens(
                _text_for_frame(row, caption_rows[str(row["filename"])])
            )
            groups: list[dict[str, Any]] = []
            matched_count = 0
            for group in target["required_token_groups"]:
                matched = [
                    token
                    for token in group["tokens"]
                    if token_matches(tokens, token, aliases)
                ]
                missing = [
                    token for token in group["tokens"] if token not in matched
                ]
                matched_count += len(matched)
                groups.append(
                    {
                        "name": group["name"],
                        "tokens": list(group["tokens"]),
                        "matched_tokens": matched,
                        "missing_tokens": missing,
                    }
                )
            candidate_results.append(
                (
                    matched_count,
                    -delta,
                    str(row["filename"]),
                    selected_timestamp,
                    matched_timestamp,
                    (
                        "selected_timestamp"
                        if math.isclose(
                            selected_timestamp,
                            matched_timestamp,
                            rel_tol=0.0,
                            abs_tol=1e-6,
                        )
                        else "merged_timestamps"
                    ),
                    groups,
                )
            )
        if not candidate_results:
            result = {
                "id": target["id"],
                "time_seconds": target["time_seconds"],
                "tolerance_seconds": target["tolerance_seconds"],
                "selected_png": None,
                "selected_timestamp": None,
                "matched_timestamp": None,
                "matched_via": None,
                "delta_seconds": None,
                "token_groups": [],
                "passed": False,
            }
            failures.append(
                f"target {target['id']} has no published frame within "
                f"{target['tolerance_seconds']:.2f} seconds"
            )
        else:
            best = max(candidate_results, key=lambda item: (item[0], item[1], item[2]))
            (
                _matched,
                negative_delta,
                name,
                selected_timestamp,
                matched_timestamp,
                matched_via,
                groups,
            ) = best
            passed = all(not group["missing_tokens"] for group in groups)
            result = {
                "id": target["id"],
                "time_seconds": target["time_seconds"],
                "tolerance_seconds": target["tolerance_seconds"],
                "selected_png": name,
                "selected_timestamp": selected_timestamp,
                "matched_timestamp": matched_timestamp,
                "matched_via": matched_via,
                "delta_seconds": -negative_delta,
                "token_groups": groups,
                "passed": passed,
            }
            if not passed:
                missing = sorted(
                    {
                        token
                        for group in groups
                        for token in group["missing_tokens"]
                    }
                )
                failures.append(
                    f"target {target['id']} is missing normalized OCR tokens: "
                    + ", ".join(missing)
                )
        results.append(result)
    return results, failures


def _trace_qualification(
    pipeline_trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    transition_rows: list[dict[str, Any]] = []
    structured_rows: list[dict[str, Any]] = []
    if pipeline_trace is not None:
        records = pipeline_trace.get("records")
        if isinstance(records, list):
            for raw_record in records:
                if not isinstance(raw_record, dict):
                    continue
                payload = raw_record.get("payload")
                if not isinstance(payload, dict):
                    continue
                candidates = payload.get("candidates")
                if not isinstance(candidates, list):
                    continue
                for raw_candidate in candidates:
                    if not isinstance(raw_candidate, dict):
                        continue
                    if (
                        raw_candidate.get("proposal_lane") == "transition"
                        and raw_candidate.get("transition_side") in {"pre", "post"}
                    ):
                        transition_rows.append(
                            {
                                "stage": raw_record.get("stage"),
                                "timestamp": raw_candidate.get("timestamp"),
                                "transition_side": raw_candidate.get("transition_side"),
                            }
                        )
                    categories = raw_candidate.get("structured_delta_categories")
                    if isinstance(categories, list) and categories:
                        structured_rows.append(
                            {
                                "stage": raw_record.get("stage"),
                                "timestamp": raw_candidate.get("timestamp"),
                                "categories": sorted(
                                    str(value) for value in categories
                                ),
                            }
                        )
    return {
        "transition_side_proposal_observed": bool(transition_rows),
        "structured_form_state_observed": bool(structured_rows),
        "transition_examples": transition_rows[:3],
        "structured_examples": structured_rows[:3],
    }


def _collect_artifact_evidence(
    bundle: Path,
    contract: FixtureContract,
    *,
    stored_paths: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], list[str]]:
    if stored_paths is None:
        manifest_relative = "artifacts/frames/manifest.json"
        captions_relative = "artifacts/frames/captions.json"
        trace_relatives = {
            "pipeline_trace": "artifacts/frames/pipeline_trace.json",
            "debug_qa_trace": "artifacts/frames/debug_qa_trace.json",
        }
    else:
        manifest_relative = _mapping(
            stored_paths.get("manifest"), "evidence artifacts.manifest"
        ).get("path")
        captions_relative = _mapping(
            stored_paths.get("captions"), "evidence artifacts.captions"
        ).get("path")
        raw_traces = stored_paths.get("traces")
        if not isinstance(raw_traces, list):
            raise ReleaseEvidenceError("evidence artifacts.traces must be a list")
        trace_relatives = {}
        for index, raw_trace in enumerate(raw_traces):
            trace = _mapping(raw_trace, f"evidence trace {index}")
            name = trace.get("name")
            if name not in {"pipeline_trace", "debug_qa_trace"}:
                raise ReleaseEvidenceError(
                    f"evidence trace {index} has an unsupported name"
                )
            if name in trace_relatives:
                raise ReleaseEvidenceError(
                    f"evidence contains duplicate trace {name!r}"
                )
            trace_relatives[name] = trace.get("path")

    manifest_path = _resolve_bundle_file(
        bundle,
        manifest_relative,
        "frame manifest",
    )
    captions_path = _resolve_bundle_file(
        bundle,
        captions_relative,
        "frame captions",
    )
    manifest = _mapping(_load_json(manifest_path, "frame manifest"), "frame manifest")
    captions = _load_json(captions_path, "frame captions")
    manifest_rows, caption_rows = _frame_rows(manifest, captions)

    png_records: list[dict[str, Any]] = []
    for row in sorted(manifest_rows, key=lambda item: str(item["filename"])):
        relative = (
            PurePosixPath(_relative_path(manifest_relative, "frame manifest path")).parent
            / str(row["filename"])
        ).as_posix()
        png_path = _resolve_bundle_file(bundle, relative, f"frame PNG {row['filename']}")
        dimensions = _png_dimensions(png_path)
        if dimensions != (1280, 720):
            raise ReleaseEvidenceError(
                f"frame PNG {row['filename']} must be 1280x720"
            )
        png_records.append(
            {
                **_artifact_file_record(png_path, bundle),
                "name": str(row["filename"]),
                "width": dimensions[0],
                "height": dimensions[1],
            }
        )
    canonical_rows = [
        {
            "name": record["name"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in png_records
    ]

    trace_records: list[dict[str, Any]] = []
    pipeline_trace: dict[str, Any] | None = None
    for name, relative in sorted(trace_relatives.items()):
        trace_path = _resolve_bundle_file(bundle, relative, f"{name} artifact")
        trace_payload = _mapping(_load_json(trace_path, name), name)
        if name == "pipeline_trace":
            pipeline_trace = trace_payload
        trace_records.append(
            {
                "name": name,
                **_artifact_file_record(trace_path, bundle),
            }
        )

    target_results, failures = _evaluate_targets(
        contract.metadata,
        manifest_rows,
        caption_rows,
    )
    budgets = _validate_budgets(contract.metadata.get("budgets"))
    frame_count = len(png_records)
    budget_failures: list[str] = []
    if frame_count < budgets["min_output_frames"]:
        budget_failures.append(
            f"published frame count {frame_count} is below "
            f"{budgets['min_output_frames']}"
        )
    if frame_count > budgets["max_output_frames"]:
        budget_failures.append(
            f"published frame count {frame_count} exceeds "
            f"{budgets['max_output_frames']}"
        )
    selected_names = [
        result["selected_png"]
        for result in target_results
        if result["selected_png"] is not None
    ]
    redundancy = (
        1.0 - (len(set(selected_names)) / len(target_results))
        if target_results
        else 1.0
    )
    if redundancy > budgets["max_target_redundancy"] + 1e-12:
        budget_failures.append(
            f"target redundancy {redundancy:.3f} exceeds "
            f"{budgets['max_target_redundancy']:.3f}"
        )

    artifacts = {
        "manifest": _artifact_file_record(manifest_path, bundle),
        "captions": _artifact_file_record(captions_path, bundle),
        "pngs": png_records,
        "canonical_png_aggregate_sha256": _canonical_json_sha256(canonical_rows),
        "traces": trace_records,
    }
    budget_result = {
        **budgets,
        "observed_output_frames": frame_count,
        "passed": not budget_failures,
    }
    redundancy_result = {
        "method": "one-minus-unique-target-artifacts-over-target-count",
        "selected_target_artifacts": len(selected_names),
        "unique_target_artifacts": len(set(selected_names)),
        "value": redundancy,
        "maximum": budgets["max_target_redundancy"],
        "passed": redundancy <= budgets["max_target_redundancy"] + 1e-12,
    }
    qualification = _trace_qualification(pipeline_trace)
    return (
        artifacts,
        target_results,
        budget_result,
        redundancy_result,
        [*failures, *budget_failures],
    )


_RUNTIME_PROBE = r"""
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

import keyframe

names = json.loads(sys.argv[1])
packages = {}
for output_name, distribution_name in names.items():
    try:
        packages[output_name] = metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        packages[output_name] = None

try:
    distribution = metadata.distribution("keyframe")
except metadata.PackageNotFoundError:
    distribution_root = None
    distribution_version = None
    direct_url = None
else:
    distribution_root = str(Path(distribution.locate_file("")).resolve())
    distribution_version = distribution.version
    direct_url_text = distribution.read_text("direct_url.json")
    try:
        direct_url = json.loads(direct_url_text) if direct_url_text else None
    except json.JSONDecodeError:
        direct_url = {"invalid": True}

print(json.dumps({
    "interpreter": sys.executable,
    "environment_root": sys.prefix,
    "base_prefix": sys.base_prefix,
    "python": platform.python_version(),
    "system": platform.system(),
    "machine": platform.machine(),
    "platform_release": platform.release(),
    "keyframe_module_path": str(Path(keyframe.__file__).resolve()),
    "keyframe_package_root": str(Path(keyframe.__file__).resolve().parent),
    "distribution_root": distribution_root,
    "distribution_version": distribution_version,
    "direct_url": direct_url,
    "packages": packages,
}))
"""


def _sanitized_environment(*, isolated: bool) -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if isolated:
        environment["PYTHONSAFEPATH"] = "1"
    else:
        environment.pop("PYTHONSAFEPATH", None)
    return environment


def probe_runtime(
    runtime_python: str | Path,
    *,
    working_directory: Path,
    isolated: bool,
) -> dict[str, Any]:
    executable = Path(
        os.path.abspath(os.path.expanduser(os.fspath(runtime_python)))
    )
    if executable.is_symlink():
        # Virtual-environment launchers are commonly symlinks. The path itself
        # is recorded and checked against the environment; the target may be a
        # shared base interpreter.
        pass
    if not executable.is_file():
        raise ReleaseEvidenceError(f"runtime Python does not exist: {executable}")
    command = [str(executable)]
    if isolated:
        command.append("-I")
    command.extend(
        [
            "-c",
            _RUNTIME_PROBE,
            json.dumps(RUNTIME_PACKAGE_NAMES, sort_keys=True),
        ]
    )
    completed = subprocess.run(
        command,
        cwd=working_directory,
        env=_sanitized_environment(isolated=isolated),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ReleaseEvidenceError(f"runtime package probe failed: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceError(
            "runtime package probe returned invalid JSON"
        ) from exc
    return _mapping(payload, "runtime package probe")


def _canonical_project_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def artifact_name_version(path: str | Path) -> tuple[str, str]:
    artifact = Path(path)
    name = artifact.name
    if name.endswith(".whl"):
        parts = name[:-4].split("-")
        if len(parts) < 5:
            raise ReleaseEvidenceError(f"invalid wheel filename: {name}")
        project_name, version = parts[0], parts[1]
    else:
        suffix = next(
            (candidate for candidate in (".tar.gz", ".tar.bz2", ".zip") if name.endswith(candidate)),
            None,
        )
        if suffix is None:
            raise ReleaseEvidenceError(
                "release artifact must be a wheel or source distribution"
            )
        stem = name[: -len(suffix)]
        match = re.fullmatch(r"(.+)-([0-9][A-Za-z0-9.!+_-]*)", stem)
        if match is None:
            raise ReleaseEvidenceError(f"invalid source distribution filename: {name}")
        project_name, version = match.groups()
    canonical = _canonical_project_name(project_name)
    if canonical != "keyframe":
        raise ReleaseEvidenceError(
            f"release artifact project name must be 'keyframe', found {canonical!r}"
        )
    if not version:
        raise ReleaseEvidenceError("release artifact version must be nonempty")
    return canonical, version


def _path_inside(path: str | Path, root: str | Path) -> bool:
    try:
        return Path(path).resolve(strict=True).is_relative_to(
            Path(root).resolve(strict=True)
        )
    except (OSError, RuntimeError):
        return False


def _validate_artifact_runtime(
    probe: Mapping[str, Any],
    *,
    artifact_name: str,
    artifact_version: str,
    artifact_filename: str,
    artifact_sha256: str,
    repository_root: Path | None,
    working_directory: Path,
) -> dict[str, Any]:
    system = probe.get("system")
    machine = str(probe.get("machine", "")).lower()
    if (system, machine) not in SUPPORTED_PLATFORMS:
        raise ReleaseEvidenceError(
            "release-frame evidence is supported only on Darwin ARM64 and "
            f"Linux x86-64; found {system} {probe.get('machine')}"
        )
    prefix = Path(str(probe.get("environment_root", ""))).expanduser()
    base_prefix = Path(str(probe.get("base_prefix", ""))).expanduser()
    if prefix == base_prefix or not (prefix / "pyvenv.cfg").is_file():
        raise ReleaseEvidenceError(
            "artifact evidence requires a dedicated virtual environment"
        )
    interpreter = Path(str(probe.get("interpreter", ""))).expanduser()
    if not interpreter.is_absolute() or not interpreter.is_relative_to(prefix):
        raise ReleaseEvidenceError(
            "artifact runtime interpreter must be launched from its environment"
        )
    package_root = probe.get("keyframe_package_root")
    distribution_root = probe.get("distribution_root")
    if not isinstance(package_root, str) or not _path_inside(package_root, prefix):
        raise ReleaseEvidenceError(
            "loaded keyframe package is outside the clean environment"
        )
    if not isinstance(distribution_root, str) or not _path_inside(
        distribution_root, prefix
    ):
        raise ReleaseEvidenceError(
            "loaded keyframe distribution is outside the clean environment"
        )
    if repository_root is not None and _path_inside(package_root, repository_root):
        raise ReleaseEvidenceError(
            "repository source shadowed the artifact-backed keyframe package"
        )
    if repository_root is not None and _path_inside(
        working_directory,
        repository_root,
    ):
        raise ReleaseEvidenceError(
            "artifact evidence working directory must be outside the checkout"
        )
    packages = _mapping(probe.get("packages"), "runtime packages")
    installed_version = packages.get("keyframe")
    if installed_version != artifact_version:
        raise ReleaseEvidenceError(
            f"installed keyframe version {installed_version!r} does not match "
            f"artifact version {artifact_version!r}"
        )
    if probe.get("distribution_version") != artifact_version:
        raise ReleaseEvidenceError(
            "loaded keyframe distribution metadata does not match the artifact"
        )
    if artifact_name != "keyframe":
        raise ReleaseEvidenceError("loaded artifact project name must be keyframe")
    direct_url = probe.get("direct_url")
    if not isinstance(direct_url, dict):
        raise ReleaseEvidenceError(
            "installed distribution is missing direct artifact provenance"
        )
    archive_info = direct_url.get("archive_info")
    if not isinstance(archive_info, dict):
        raise ReleaseEvidenceError(
            "installed distribution does not identify its source archive"
        )
    hashes = archive_info.get("hashes")
    recorded_hash = (
        hashes.get("sha256")
        if isinstance(hashes, dict)
        else None
    )
    if recorded_hash is None:
        legacy_hash = archive_info.get("hash")
        if (
            isinstance(legacy_hash, str)
            and legacy_hash.startswith("sha256=")
        ):
            recorded_hash = legacy_hash.removeprefix("sha256=")
    if recorded_hash != artifact_sha256:
        raise ReleaseEvidenceError(
            "installed distribution archive hash does not match the supplied artifact"
        )
    direct_url_value = direct_url.get("url")
    if not isinstance(direct_url_value, str):
        raise ReleaseEvidenceError(
            "installed distribution archive URL is missing"
        )
    archive_filename = Path(
        urllib.parse.unquote(
            urllib.parse.urlparse(direct_url_value).path
        )
    ).name
    if archive_filename != artifact_filename:
        raise ReleaseEvidenceError(
            "installed distribution archive name does not match the supplied artifact"
        )
    return {
        "interpreter": str(interpreter),
        "environment_root": str(prefix.resolve()),
        "base_prefix": str(base_prefix.resolve()),
        "working_directory": str(working_directory.resolve()),
        "keyframe_module_path": str(probe["keyframe_module_path"]),
        "keyframe_package_root": str(Path(package_root).resolve()),
        "distribution_root": str(Path(distribution_root).resolve()),
        "isolated": True,
        "repository_pythonpath_present": False,
        "repository_shadowing": False,
        "package_inside_environment": True,
        "distribution_inside_environment": True,
        "installation_archive_filename": artifact_filename,
        "installation_archive_sha256": artifact_sha256,
    }


def _git(
    repository: Path,
    *arguments: str,
) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ReleaseEvidenceError(f"Git source identity failed: {detail}")
    return completed.stdout.strip()


def _source_identity_from_probe(
    probe: Mapping[str, Any],
    *,
    source_tree: Path,
    working_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package_root = Path(str(probe.get("keyframe_package_root", ""))).resolve()
    candidate_repository = Path(
        _git(package_root, "rev-parse", "--show-toplevel")
    ).resolve()
    if not package_root.is_relative_to(candidate_repository):
        raise ReleaseEvidenceError(
            "loaded keyframe package is unrelated to its derived Git repository"
        )
    if candidate_repository != source_tree.resolve():
        raise ReleaseEvidenceError(
            "loaded keyframe package did not come from the supplied source tree"
        )
    dirty = _git(
        candidate_repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if dirty:
        raise ReleaseEvidenceError(
            "dirty Git source cannot produce release-frame evidence"
        )
    commit = _git(candidate_repository, "rev-parse", "HEAD")
    if not COMMIT_RE.fullmatch(commit):
        raise ReleaseEvidenceError("Git source identity did not resolve a full commit")
    pyproject = candidate_repository / "pyproject.toml"
    try:
        version = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
            "version"
        ]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseEvidenceError(
            "could not resolve source package version from pyproject.toml"
        ) from exc
    if not isinstance(version, str) or not version:
        raise ReleaseEvidenceError("source package version must be nonempty")
    source_identity = {
        "kind": "git",
        "commit_sha": commit,
        "version": version,
    }
    locations = {
        "interpreter": str(probe.get("interpreter")),
        "environment_root": str(probe.get("environment_root")),
        "base_prefix": str(probe.get("base_prefix")),
        "working_directory": str(working_directory.resolve()),
        "keyframe_module_path": str(probe.get("keyframe_module_path")),
        "keyframe_package_root": str(package_root),
        "distribution_root": None,
        "isolated": False,
        "repository_pythonpath_present": False,
        "repository_shadowing": False,
        "package_inside_environment": None,
        "distribution_inside_environment": None,
    }
    return source_identity, locations


def _model_provenance(probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    system = str(probe.get("system"))
    machine = str(probe.get("machine", "")).lower()
    ocr_id = (
        "com.apple.Vision.VNRecognizeTextRequest"
        if (system, machine) == ("Darwin", "arm64")
        else "PaddleOCR/default-English-pipeline"
    )
    return [
        {
            "role": "visual-embedding",
            "model_id": "open_clip/ViT-B-32/laion2b_s34b_b79k",
            "repository_revision": None,
            "stable_weight_files": [],
        },
        {
            "role": "captioning",
            "model_id": "florence-community/Florence-2-base",
            "repository_revision": None,
            "stable_weight_files": [],
        },
        {
            "role": "ocr",
            "model_id": ocr_id,
            "repository_revision": None,
            "stable_weight_files": [],
        },
    ]


def _platform_record(probe: Mapping[str, Any]) -> dict[str, Any]:
    system = probe.get("system")
    machine = str(probe.get("machine", "")).lower()
    if (system, machine) not in SUPPORTED_PLATFORMS:
        raise ReleaseEvidenceError(
            f"unsupported evidence platform: {system} {probe.get('machine')}"
        )
    return {
        "system": system,
        "machine": machine,
        "platform_release": probe.get("platform_release"),
        "python": probe.get("python"),
    }


def _ocr_record(probe: Mapping[str, Any]) -> dict[str, Any]:
    system = str(probe.get("system"))
    machine = str(probe.get("machine", "")).lower()
    backend = SUPPORTED_PLATFORMS[(system, machine)]
    packages = _mapping(probe.get("packages"), "runtime packages")
    if backend == "apple-vision":
        return {
            "backend": backend,
            "framework": "Vision",
            "package_version": packages.get("pyobjc_framework_vision"),
        }
    return {
        "backend": backend,
        "framework": "PaddleOCR",
        "package_version": packages.get("paddleocr"),
        "engine_version": packages.get("paddlepaddle"),
    }


def _copy_fixture(contract: FixtureContract, bundle: Path) -> FixtureContract:
    destination = bundle / "fixture"
    destination.mkdir(parents=True)
    for path in (
        contract.metadata_path,
        contract.recording_path,
        contract.license_path,
        *contract.source_paths,
    ):
        shutil.copy2(path, destination / path.name)
    return load_fixture_contract(destination / contract.metadata_path.name)


def _copy_artifact(artifact: Path, bundle: Path) -> Path:
    destination = bundle / "source"
    destination.mkdir(parents=True)
    target = destination / artifact.name
    shutil.copy2(artifact, target)
    return target


def _run_default_cli(
    *,
    runtime_python: Path,
    fixture: Path,
    output_dir: Path,
    qa_targets: Path,
    working_directory: Path,
    isolated: bool,
) -> list[str]:
    command = [str(runtime_python)]
    if isolated:
        command.append("-I")
    command.extend(
        [
            "-m",
            "keyframe.cli",
            str(fixture),
            "--output",
            str(output_dir),
            "--frames-only",
            "--sample-interval",
            str(DEFAULT_CONFIGURATION["sample_interval_seconds"]),
            "--pass1-clusters",
            str(DEFAULT_CONFIGURATION["pass1_clusters"]),
            "--max-output-frames",
            str(DEFAULT_CONFIGURATION["max_output_frames"]),
            "--verbose-trace",
            "--debug-qa-targets",
            str(qa_targets),
        ]
    )
    completed = subprocess.run(
        command,
        cwd=working_directory,
        env=_sanitized_environment(isolated=isolated),
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ReleaseEvidenceError(
            f"default Keyframe CLI route failed with exit {completed.returncode}"
        )
    return command


def _configuration_record(qa_target_path: Path, bundle: Path) -> dict[str, Any]:
    return {
        **DEFAULT_CONFIGURATION,
        "qa_targets": _artifact_file_record(qa_target_path, bundle),
    }


def run_fresh(
    *,
    fixture: str | Path,
    metadata: str | Path,
    output: str | Path,
    runtime_python: str | Path,
    artifact: str | Path | None = None,
    source_tree: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the default frame CLI and create a replayable evidence bundle."""

    if (artifact is None) == (source_tree is None):
        raise ReleaseEvidenceError(
            "fresh evidence requires exactly one of artifact or source_tree"
        )
    original_contract = load_fixture_contract(metadata, recording_path=fixture)
    bundle = _prepare_empty_directory(Path(output).expanduser(), "evidence output")
    contract = _copy_fixture(original_contract, bundle)
    configuration_dir = bundle / "configuration"
    configuration_dir.mkdir()
    qa_target_path = configuration_dir / "qa-targets.json"
    _atomic_write_json(qa_target_path, qa_targets_from_fixture(contract.metadata))

    repository = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else None
    )
    runtime = Path(
        os.path.abspath(os.path.expanduser(os.fspath(runtime_python)))
    )
    with tempfile.TemporaryDirectory(
        prefix="keyframe-release-runtime-",
        dir="/tmp",
    ) as raw_working:
        external_working = Path(raw_working).resolve()
        isolated = artifact is not None
        probe_working = (
            external_working
            if isolated
            else Path(source_tree).expanduser().resolve()
        )
        probe = probe_runtime(
            runtime,
            working_directory=probe_working,
            isolated=isolated,
        )
        if artifact is not None:
            artifact_path = Path(artifact).expanduser()
            if artifact_path.is_symlink() or not artifact_path.is_file():
                raise ReleaseEvidenceError(
                    f"release artifact must be a regular non-symlinked file: {artifact_path}"
                )
            artifact_path = artifact_path.resolve()
            artifact_name, artifact_version = artifact_name_version(artifact_path)
            artifact_hash = _sha256(artifact_path)
            locations = _validate_artifact_runtime(
                probe,
                artifact_name=artifact_name,
                artifact_version=artifact_version,
                artifact_filename=artifact_path.name,
                artifact_sha256=artifact_hash,
                repository_root=repository,
                working_directory=external_working,
            )
            bundled_artifact = _copy_artifact(artifact_path, bundle)
            source_identity = {
                "kind": "artifact",
                "name": artifact_name,
                "version": artifact_version,
                "filename": bundled_artifact.name,
                "path": _portable_relative(bundled_artifact, bundle),
                "size_bytes": bundled_artifact.stat().st_size,
                "sha256": artifact_hash,
            }
        else:
            source = Path(source_tree).expanduser().resolve()
            source_identity, locations = _source_identity_from_probe(
                probe,
                source_tree=source,
                working_directory=probe_working,
            )

        artifact_output = bundle / "artifacts"
        artifact_output.mkdir()
        _run_default_cli(
            runtime_python=runtime,
            fixture=contract.recording_path,
            output_dir=artifact_output,
            qa_targets=qa_target_path,
            working_directory=external_working if isolated else probe_working,
            isolated=isolated,
        )

    (
        artifacts,
        targets,
        budgets,
        redundancy,
        failures,
    ) = _collect_artifact_evidence(bundle, contract)
    report = {
        "identifier": EVIDENCE_IDENTIFIER,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "source_identity": source_identity,
        "fixture": _fixture_record(contract, bundle),
        "artifacts": artifacts,
        "configuration": _configuration_record(qa_target_path, bundle),
        "targets": targets,
        "budgets": budgets,
        "redundancy": redundancy,
        "platform": _platform_record(probe),
        "packages": dict(_mapping(probe.get("packages"), "runtime packages")),
        "ocr_backend": _ocr_record(probe),
        "model_provenance": _model_provenance(probe),
        "package_locations": locations,
        "qualification": _trace_qualification(
            _mapping(
                _load_json(
                    _resolve_bundle_file(
                        bundle,
                        next(
                            trace["path"]
                            for trace in artifacts["traces"]
                            if trace["name"] == "pipeline_trace"
                        ),
                        "pipeline trace",
                    ),
                    "pipeline trace",
                ),
                "pipeline trace",
            )
        ),
        "validation": {"passed": not failures, "failures": failures},
    }
    evidence_path = bundle / "evidence.json"
    _atomic_write_json(evidence_path, report)
    return report


def _without_pass_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_pass_fields(item)
            for key, item in value.items()
            if key not in {"passed", "status"}
        }
    if isinstance(value, list):
        return [_without_pass_fields(item) for item in value]
    return value


def _compare_recorded(
    label: str,
    recorded: Any,
    recomputed: Any,
    failures: list[str],
    *,
    ignore_pass_fields: bool = False,
) -> None:
    left = _without_pass_fields(recorded) if ignore_pass_fields else recorded
    right = _without_pass_fields(recomputed) if ignore_pass_fields else recomputed
    if left != right:
        failures.append(f"{label} does not match replayed bundle contents")


def _validate_source_identity(
    report: Mapping[str, Any],
    bundle: Path,
    failures: list[str],
) -> None:
    source = report.get("source_identity")
    if not isinstance(source, dict):
        failures.append("source_identity must be an object")
        return
    kind = source.get("kind")
    version = source.get("version")
    if not isinstance(version, str) or not version:
        failures.append("source_identity version must be nonempty")
    packages = report.get("packages")
    if not isinstance(packages, dict) or packages.get("keyframe") != version:
        failures.append(
            "runtime keyframe package version must match source_identity"
        )
    if kind == "git":
        commit = source.get("commit_sha")
        if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
            failures.append("Git source_identity must contain a full commit SHA")
        if set(source) != {"kind", "commit_sha", "version"}:
            failures.append("Git source_identity has unexpected fields")
        return
    if kind != "artifact":
        failures.append("source_identity kind must be git or artifact")
        return
    expected_fields = {
        "kind",
        "name",
        "version",
        "filename",
        "path",
        "size_bytes",
        "sha256",
    }
    if set(source) != expected_fields:
        failures.append("artifact source_identity has unexpected fields")
    try:
        artifact_path = _resolve_bundle_file(
            bundle,
            source.get("path"),
            "source artifact",
        )
    except ReleaseEvidenceError as exc:
        failures.append(str(exc))
        return
    try:
        name, artifact_version = artifact_name_version(artifact_path)
    except ReleaseEvidenceError as exc:
        failures.append(str(exc))
        return
    if source.get("name") != name or source.get("filename") != artifact_path.name:
        failures.append("source artifact name does not match source_identity")
    if source.get("version") != artifact_version:
        failures.append("source artifact version does not match source_identity")
    if source.get("size_bytes") != artifact_path.stat().st_size:
        failures.append("source artifact size does not match source_identity")
    if source.get("sha256") != _sha256(artifact_path):
        failures.append("source artifact hash does not match source_identity")


def _validate_platform_and_locations(
    report: Mapping[str, Any],
    failures: list[str],
) -> None:
    platform_record = report.get("platform")
    if not isinstance(platform_record, dict):
        failures.append("platform evidence must be an object")
        return
    system = platform_record.get("system")
    machine = str(platform_record.get("machine", "")).lower()
    expected_backend = SUPPORTED_PLATFORMS.get((system, machine))
    if expected_backend is None:
        failures.append("evidence platform must be Darwin ARM64 or Linux x86-64")
    ocr = report.get("ocr_backend")
    if not isinstance(ocr, dict) or ocr.get("backend") != expected_backend:
        failures.append("OCR backend does not match the evidence platform")
    source = report.get("source_identity")
    locations = report.get("package_locations")
    if not isinstance(source, dict) or not isinstance(locations, dict):
        failures.append("package location evidence must be an object")
        return
    if source.get("kind") == "artifact":
        requirements = {
            "isolated": True,
            "repository_pythonpath_present": False,
            "repository_shadowing": False,
            "package_inside_environment": True,
            "distribution_inside_environment": True,
        }
        for field, expected in requirements.items():
            if locations.get(field) is not expected:
                failures.append(
                    f"artifact package_locations.{field} must be {expected!r}"
                )
        for field in (
            "interpreter",
            "environment_root",
            "working_directory",
            "keyframe_module_path",
            "keyframe_package_root",
            "distribution_root",
            "installation_archive_filename",
            "installation_archive_sha256",
        ):
            if not isinstance(locations.get(field), str) or not locations[field]:
                failures.append(
                    f"artifact package_locations.{field} must be nonempty"
                )
        if (
            locations.get("installation_archive_filename")
            != source.get("filename")
        ):
            failures.append(
                "artifact installation archive filename must match source_identity"
            )
        if (
            locations.get("installation_archive_sha256")
            != source.get("sha256")
        ):
            failures.append(
                "artifact installation archive hash must match source_identity"
            )


def _configuration_from_report(
    report: Mapping[str, Any],
    bundle: Path,
) -> dict[str, Any]:
    configuration = _mapping(report.get("configuration"), "evidence configuration")
    qa_targets = _mapping(
        configuration.get("qa_targets"),
        "evidence configuration.qa_targets",
    )
    qa_path = _resolve_bundle_file(
        bundle,
        qa_targets.get("path"),
        "evidence QA targets",
    )
    qa_record = _artifact_file_record(qa_path, bundle)
    payload = _load_json(qa_path, "evidence QA targets")
    return {
        **DEFAULT_CONFIGURATION,
        "qa_targets": qa_record,
        "_payload": payload,
    }


def replay_bundle(bundle: str | Path) -> ReplayResult:
    """Rehash, reparse, and recompute a standalone evidence bundle."""

    root = Path(bundle).expanduser()
    failures: list[str] = []
    try:
        evidence_path = _resolve_bundle_file(root, "evidence.json", "evidence report")
        report = _mapping(_load_json(evidence_path, "evidence report"), "evidence report")
        if report.get("identifier") != EVIDENCE_IDENTIFIER:
            failures.append(
                f"evidence identifier must be {EVIDENCE_IDENTIFIER!r}"
            )
        if report.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            failures.append(
                f"evidence schema_version must be {EVIDENCE_SCHEMA_VERSION}"
            )

        fixture_record = _mapping(report.get("fixture"), "evidence fixture")
        metadata_record = _mapping(
            fixture_record.get("metadata"),
            "evidence fixture.metadata",
        )
        recording_record = _mapping(
            fixture_record.get("recording"),
            "evidence fixture.recording",
        )
        metadata_path = _resolve_bundle_file(
            root,
            metadata_record.get("path"),
            "evidence fixture metadata",
        )
        recording_path = _resolve_bundle_file(
            root,
            recording_record.get("path"),
            "evidence fixture recording",
        )
        contract = load_fixture_contract(
            metadata_path,
            recording_path=recording_path,
        )
        recomputed_fixture = _fixture_record(contract, root)

        configuration_with_payload = _configuration_from_report(report, root)
        qa_payload = configuration_with_payload.pop("_payload")
        expected_qa = qa_targets_from_fixture(contract.metadata)
        if qa_payload != expected_qa:
            failures.append(
                "evidence QA targets do not match fixture metadata"
            )
        _compare_recorded(
            "configuration",
            report.get("configuration"),
            configuration_with_payload,
            failures,
        )

        (
            recomputed_artifacts,
            recomputed_targets,
            recomputed_budgets,
            recomputed_redundancy,
            recomputed_failures,
        ) = _collect_artifact_evidence(
            root,
            contract,
            stored_paths=_mapping(report.get("artifacts"), "evidence artifacts"),
        )
        failures.extend(recomputed_failures)
        pipeline_trace = None
        for trace in recomputed_artifacts["traces"]:
            if trace["name"] == "pipeline_trace":
                pipeline_trace = _mapping(
                    _load_json(
                        _resolve_bundle_file(
                            root,
                            trace["path"],
                            "pipeline trace",
                        ),
                        "pipeline trace",
                    ),
                    "pipeline trace",
                )
        recomputed_qualification = _trace_qualification(pipeline_trace)

        _compare_recorded(
            "fixture evidence",
            report.get("fixture"),
            recomputed_fixture,
            failures,
        )
        _compare_recorded(
            "artifact evidence",
            report.get("artifacts"),
            recomputed_artifacts,
            failures,
        )
        _compare_recorded(
            "target evidence",
            report.get("targets"),
            recomputed_targets,
            failures,
            ignore_pass_fields=True,
        )
        _compare_recorded(
            "budget evidence",
            report.get("budgets"),
            recomputed_budgets,
            failures,
            ignore_pass_fields=True,
        )
        _compare_recorded(
            "redundancy evidence",
            report.get("redundancy"),
            recomputed_redundancy,
            failures,
            ignore_pass_fields=True,
        )
        _compare_recorded(
            "qualification evidence",
            report.get("qualification"),
            recomputed_qualification,
            failures,
        )
        _validate_source_identity(report, root, failures)
        _validate_platform_and_locations(report, failures)
        model_provenance = report.get("model_provenance")
        if not isinstance(model_provenance, list) or not model_provenance:
            failures.append("model_provenance must name observed models")
        else:
            for index, raw_model in enumerate(model_provenance):
                if not isinstance(raw_model, dict):
                    failures.append(f"model_provenance {index} must be an object")
                    continue
                if not isinstance(raw_model.get("model_id"), str):
                    failures.append(
                        f"model_provenance {index} model_id must be nonempty"
                    )
                weights = raw_model.get("stable_weight_files")
                if not isinstance(weights, list):
                    failures.append(
                        f"model_provenance {index} stable_weight_files must be a list"
                    )
        recomputed = {
            "fixture": recomputed_fixture,
            "artifacts": recomputed_artifacts,
            "configuration": configuration_with_payload,
            "targets": recomputed_targets,
            "budgets": recomputed_budgets,
            "redundancy": recomputed_redundancy,
            "qualification": recomputed_qualification,
        }
    except ReleaseEvidenceError as exc:
        return ReplayResult(
            report=locals().get("report"),
            recomputed=None,
            failures=(str(exc),),
        )
    return ReplayResult(
        report=report,
        recomputed=recomputed,
        failures=tuple(dict.fromkeys(failures)),
    )


def referenced_bundle_files(report: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the complete declared regular-file set for a validated bundle."""

    paths = {"evidence.json"}
    fixture = _mapping(report.get("fixture"), "evidence fixture")
    for field in ("metadata", "recording"):
        paths.add(
            str(_mapping(fixture.get(field), f"evidence fixture.{field}")["path"])
        )
    supporting = fixture.get("supporting_files")
    if not isinstance(supporting, list):
        raise ReleaseEvidenceError(
            "evidence fixture.supporting_files must be a list"
        )
    for index, record in enumerate(supporting):
        paths.add(
            str(
                _mapping(
                    record,
                    f"evidence fixture supporting file {index}",
                )["path"]
            )
        )
    artifacts = _mapping(report.get("artifacts"), "evidence artifacts")
    for field in ("manifest", "captions"):
        paths.add(
            str(_mapping(artifacts.get(field), f"evidence artifacts.{field}")["path"])
        )
    for collection in ("pngs", "traces"):
        records = artifacts.get(collection)
        if not isinstance(records, list):
            raise ReleaseEvidenceError(
                f"evidence artifacts.{collection} must be a list"
            )
        for index, record in enumerate(records):
            paths.add(
                str(
                    _mapping(
                        record,
                        f"evidence artifacts.{collection} {index}",
                    )["path"]
                )
            )
    configuration = _mapping(report.get("configuration"), "evidence configuration")
    paths.add(
        str(
            _mapping(
                configuration.get("qa_targets"),
                "evidence configuration.qa_targets",
            )["path"]
        )
    )
    source = _mapping(report.get("source_identity"), "source_identity")
    if source.get("kind") == "artifact":
        paths.add(str(source["path"]))
    for path in paths:
        _relative_path(path, "evidence file path")
    return tuple(sorted(paths))


def copy_validated_bundle(
    source: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Copy only the regular files declared by a passing evidence bundle."""

    source_root = Path(source).expanduser()
    replay = replay_bundle(source_root)
    if not replay.passed or replay.report is None:
        raise ReleaseEvidenceError(
            "source evidence bundle failed replay: " + "; ".join(replay.failures)
        )
    destination_root = _prepare_empty_directory(
        Path(destination).expanduser(),
        "evidence copy destination",
    )
    for relative in referenced_bundle_files(replay.report):
        source_path = _resolve_bundle_file(
            source_root,
            relative,
            f"evidence copy source {relative}",
        )
        target = destination_root / PurePosixPath(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
    copied = replay_bundle(destination_root)
    if not copied.passed or copied.report is None:
        raise ReleaseEvidenceError(
            "copied evidence bundle failed replay: " + "; ".join(copied.failures)
        )
    return copied.report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or replay public Keyframe frame-fixture evidence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fresh = subparsers.add_parser("fresh")
    fresh.add_argument("--fixture", required=True)
    fresh.add_argument("--metadata", required=True)
    fresh.add_argument("--output", required=True)
    fresh.add_argument("--runtime-python", default=sys.executable)
    source = fresh.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact")
    source.add_argument("--source-tree")
    fresh.add_argument("--repository-root")

    replay = subparsers.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "fresh":
            report = run_fresh(
                fixture=args.fixture,
                metadata=args.metadata,
                output=args.output,
                runtime_python=args.runtime_python,
                artifact=args.artifact,
                source_tree=args.source_tree,
                repository_root=args.repository_root,
            )
            validation = report["validation"]
            print(
                f"Evidence report: "
                f"{(Path(args.output).expanduser() / 'evidence.json').resolve()}"
            )
        else:
            replay = replay_bundle(args.bundle)
            validation = replay.validation()
    except ReleaseEvidenceError as exc:
        print(f"Release evidence error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(validation, indent=2))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
