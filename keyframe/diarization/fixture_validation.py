"""Fixture validation gate for diarization benchmark inputs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from keyframe.diarization.bundles import validate_candidate_bundle_payload
from keyframe.diarization.io import recording_from_dict
from keyframe.diarization.manifests import DatasetManifest, ScoringPolicyManifest
from keyframe.diarization.models import CanonicalRecording, ValidationError
from keyframe.diarization.provenance import (
    AudioTransformConfig,
    AudioTransformManifest,
    hash_audio_transform_config,
    sha256_file,
)
from keyframe.diarization.scoring_exports import validate_rttm_text, validate_uem_text


FixtureValidationStatus = Literal["valid", "invalid_fixture"]
FixtureIssueCategory = Literal[
    "audio_metadata_mismatch",
    "checksum_mismatch",
    "invalid_interval",
    "missing_file",
    "missing_scoring_export",
    "reference_leakage",
    "schema_validation",
    "transform_config_mismatch",
    "unsupported_overlap",
    "unresolved_speaker",
]
FixtureSliceStatus = Literal["ready", "insufficient_support"]

_ISSUE_CATEGORIES = frozenset(
    {
        "audio_metadata_mismatch",
        "checksum_mismatch",
        "invalid_interval",
        "missing_file",
        "missing_scoring_export",
        "reference_leakage",
        "schema_validation",
        "transform_config_mismatch",
        "unsupported_overlap",
        "unresolved_speaker",
    }
)
_VALIDATION_STATUSES = frozenset({"valid", "invalid_fixture"})
_SLICE_STATUSES = frozenset({"ready", "insufficient_support"})


@dataclass(frozen=True)
class FixtureValidationIssue:
    """One fixture problem that should block benchmark execution."""

    issue_id: str
    category: FixtureIssueCategory
    message: str
    path: str | None = None
    recording_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "issue_id", _require_id(self.issue_id, "fixture_issue.issue_id"))
        category = _require_id(self.category, "fixture_issue.category")
        if category not in _ISSUE_CATEGORIES:
            raise ValidationError(f"fixture_issue.category is not supported: {category}")
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "message", _require_text(self.message, "fixture_issue.message"))
        object.__setattr__(self, "path", _optional_text(self.path, "fixture_issue.path"))
        object.__setattr__(self, "recording_id", _optional_text(self.recording_id, "fixture_issue.recording_id"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FixtureSliceMetadata:
    """Slice status emitted by validation for later report rendering."""

    slice_id: str
    dimension: str
    value: str
    status: FixtureSliceStatus
    support_count: int
    minimum_support: int
    recording_ids: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slice_id", _require_id(self.slice_id, "fixture_slice.slice_id"))
        object.__setattr__(self, "dimension", _require_id(self.dimension, "fixture_slice.dimension"))
        object.__setattr__(self, "value", _require_id(self.value, "fixture_slice.value"))
        status = _require_id(self.status, "fixture_slice.status")
        if status not in _SLICE_STATUSES:
            raise ValidationError(f"fixture_slice.status is not supported: {status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "support_count",
            _require_non_negative_int(self.support_count, "fixture_slice.support_count"),
        )
        object.__setattr__(
            self,
            "minimum_support",
            _require_positive_int(self.minimum_support, "fixture_slice.minimum_support"),
        )
        recording_ids = _unique_tuple_of_ids(self.recording_ids, "fixture_slice.recording_ids")
        object.__setattr__(self, "recording_ids", recording_ids)
        if self.support_count != len(recording_ids):
            raise ValidationError("fixture_slice.support_count must match recording_ids")
        if self.status == "ready" and self.support_count < self.minimum_support:
            raise ValidationError("ready fixture slices must meet minimum support")
        if self.status == "insufficient_support" and self.support_count >= self.minimum_support:
            raise ValidationError("insufficient fixture slices must be below minimum support")
        object.__setattr__(self, "metrics", _validate_json_object(self.metrics, "fixture_slice.metrics"))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recording_ids"] = list(self.recording_ids)
        payload["metrics"] = _thaw_json_value(self.metrics)
        return payload


@dataclass(frozen=True)
class FixtureValidationResult:
    """Aggregated fixture validation result before benchmark execution."""

    status: FixtureValidationStatus
    issues: tuple[FixtureValidationIssue, ...] = ()
    slice_metadata: tuple[FixtureSliceMetadata, ...] = ()
    checked_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        status = _require_id(self.status, "fixture_validation.status")
        if status not in _VALIDATION_STATUSES:
            raise ValidationError(f"fixture_validation.status is not supported: {status}")
        object.__setattr__(self, "status", status)
        issues = _tuple_of(self.issues, FixtureValidationIssue, "fixture_validation.issues")
        object.__setattr__(self, "issues", issues)
        object.__setattr__(
            self,
            "slice_metadata",
            _tuple_of(self.slice_metadata, FixtureSliceMetadata, "fixture_validation.slice_metadata"),
        )
        object.__setattr__(
            self,
            "checked_files",
            _unique_tuple_of_paths(self.checked_files, "fixture_validation.checked_files"),
        )
        if self.status == "valid" and issues:
            raise ValidationError("valid fixture results cannot include issues")
        if self.status == "invalid_fixture" and not issues:
            raise ValidationError("invalid_fixture results must include issues")

    @property
    def valid(self) -> bool:
        return self.status == "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_files": list(self.checked_files),
            "issues": [issue.to_dict() for issue in self.issues],
            "slice_metadata": [item.to_dict() for item in self.slice_metadata],
            "status": self.status,
        }


@dataclass(frozen=True)
class AudioTransformCacheCheck:
    """Fixture-gate input for validating cached canonical audio transform artifacts."""

    manifest: AudioTransformManifest
    canonical_audio_path: str | Path | None = None
    original_audio_path: str | Path | None = None
    expected_config: AudioTransformConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, AudioTransformManifest):
            raise ValidationError("audio_transform_cache.manifest must be an AudioTransformManifest")
        object.__setattr__(
            self,
            "canonical_audio_path",
            _optional_path_text(self.canonical_audio_path, "audio_transform_cache.canonical_audio_path"),
        )
        object.__setattr__(
            self,
            "original_audio_path",
            _optional_path_text(self.original_audio_path, "audio_transform_cache.original_audio_path"),
        )
        if self.expected_config is not None and not isinstance(self.expected_config, AudioTransformConfig):
            raise ValidationError("audio_transform_cache.expected_config must be an AudioTransformConfig")

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_audio_path": self.canonical_audio_path,
            "expected_config": None if self.expected_config is None else self.expected_config.to_dict(),
            "manifest": self.manifest.to_dict(),
            "original_audio_path": self.original_audio_path,
        }


def validate_manifest_expected_files(
    manifest: DatasetManifest,
    *,
    root: str | Path = ".",
) -> FixtureValidationResult:
    """Validate manifest-declared files, sizes, and checksums."""

    if not isinstance(manifest, DatasetManifest):
        raise ValidationError("manifest must be a DatasetManifest")
    root = Path(root)
    checked_files: list[str] = []
    issues: list[FixtureValidationIssue] = []
    for expected_file in manifest.expected_files:
        path = root / expected_file.path
        path_text = path.as_posix()
        checked_files.append(path_text)
        if not path.is_file():
            issues.append(
                _issue(
                    "missing_file",
                    f"missing expected fixture file: {expected_file.path}",
                    path=path_text,
                )
            )
            continue
        if expected_file.size_bytes is not None and path.stat().st_size != expected_file.size_bytes:
            issues.append(
                _issue(
                    "checksum_mismatch",
                    f"fixture file size mismatch: {expected_file.path}",
                    path=path_text,
                )
            )
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_file.checksum_sha256:
            issues.append(
                _issue(
                    "checksum_mismatch",
                    f"fixture file checksum mismatch: {expected_file.path}",
                    path=path_text,
                )
            )
    return _result(issues, checked_files=tuple(checked_files))


def validate_audio_transform_cache(
    manifest: AudioTransformManifest,
    *,
    canonical_audio_path: str | Path | None = None,
    original_audio_path: str | Path | None = None,
    expected_config: AudioTransformConfig | None = None,
) -> FixtureValidationResult:
    """Validate content-addressed audio transform cache bytes and config hash."""

    if not isinstance(manifest, AudioTransformManifest):
        raise ValidationError("manifest must be an AudioTransformManifest")
    if expected_config is not None and not isinstance(expected_config, AudioTransformConfig):
        raise ValidationError("expected_config must be an AudioTransformConfig")

    issues: list[FixtureValidationIssue] = []
    checked_files: list[str] = []
    if expected_config is not None and hash_audio_transform_config(expected_config) != manifest.transform_config_hash:
        issues.append(
            _issue(
                "transform_config_mismatch",
                "audio transform config hash does not match cached manifest",
            )
        )
    if original_audio_path is not None:
        checked_files.append(
            _validate_audio_cache_file(
                original_audio_path,
                expected_sha256=manifest.original_audio_sha256,
                label="original audio",
                issues=issues,
            )
        )
    if canonical_audio_path is not None:
        checked_files.append(
            _validate_audio_cache_file(
                canonical_audio_path,
                expected_sha256=manifest.canonical_audio_sha256,
                label="canonical audio",
                issues=issues,
            )
        )
    return _result(issues, checked_files=tuple(checked_files))


def validate_canonical_reference_payload(
    payload: dict[str, Any],
    *,
    scoring_policy: ScoringPolicyManifest | None = None,
    minimum_slice_support: int = 1,
) -> FixtureValidationResult:
    """Validate a serialized canonical reference without raising fixture errors."""

    try:
        recording = recording_from_dict(payload)
    except ValidationError as exc:
        return _result((_issue(_category_for_model_error(str(exc)), str(exc)),))
    return validate_canonical_recording(
        recording,
        scoring_policy=scoring_policy,
        minimum_slice_support=minimum_slice_support,
    )


def validate_canonical_recording(
    recording: CanonicalRecording,
    *,
    scoring_policy: ScoringPolicyManifest | None = None,
    minimum_slice_support: int = 1,
) -> FixtureValidationResult:
    """Validate canonical reference semantics and policy-dependent overlap handling."""

    if not isinstance(recording, CanonicalRecording):
        raise ValidationError("recording must be a CanonicalRecording")
    return _result(
        _canonical_recording_issues(recording, scoring_policy=scoring_policy),
        slice_metadata=build_fixture_slice_metadata((recording,), minimum_support=minimum_slice_support),
    )


def _canonical_recording_issues(
    recording: CanonicalRecording,
    *,
    scoring_policy: ScoringPolicyManifest | None,
) -> tuple[FixtureValidationIssue, ...]:
    issues: list[FixtureValidationIssue] = []

    for word in recording.words:
        if word.speaker_ref is None:
            issues.append(
                _issue(
                    "unresolved_speaker",
                    f"word {word.word_id} has no resolved speaker_ref",
                    recording_id=recording.recording_id,
                )
            )

    has_overlap = (
        any(word.overlap for word in recording.words)
        or any(span.overlap for span in recording.speaker_spans)
        or _has_overlapping_speaker_spans(recording.speaker_spans)
    )
    if has_overlap and (scoring_policy is None or scoring_policy.ignore_overlap):
        issues.append(
            _issue(
                "unsupported_overlap",
                "overlap requires a scoring policy that includes overlap",
                recording_id=recording.recording_id,
            )
        )

    return tuple(issues)


def validate_candidate_bundle_against_reference(
    payload: dict[str, Any],
    recording: CanonicalRecording,
    *,
    allow_mono_mix: bool = False,
) -> FixtureValidationResult:
    """Validate candidate bundle redaction and candidate-visible audio metadata."""

    if not isinstance(recording, CanonicalRecording):
        raise ValidationError("recording must be a CanonicalRecording")
    try:
        validate_candidate_bundle_payload(payload)
    except ValidationError as exc:
        return _result((_issue(_category_for_candidate_error(str(exc)), str(exc), recording_id=recording.recording_id),))

    issues: list[FixtureValidationIssue] = []
    audio = payload["audio"]
    channels = payload["channels"]
    timeline = payload["runtime_hints"]["timeline"]
    if audio["duration_ms"] != recording.duration_ms:
        issues.append(
            _issue(
                "audio_metadata_mismatch",
                "candidate audio duration_ms does not match canonical reference",
                recording_id=recording.recording_id,
            )
        )
    if audio["sample_rate_hz"] != recording.sample_rate_hz:
        issues.append(
            _issue(
                "audio_metadata_mismatch",
                "candidate audio sample_rate_hz does not match canonical reference",
                recording_id=recording.recording_id,
            )
        )
    if audio["time_basis"] != recording.time_basis:
        issues.append(
            _issue(
                "audio_metadata_mismatch",
                "candidate audio time_basis does not match canonical reference",
                recording_id=recording.recording_id,
            )
        )
    if timeline["timeline_id"] != recording.timeline_id:
        issues.append(
            _issue(
                "audio_metadata_mismatch",
                "candidate audio timeline_id does not match canonical reference",
                recording_id=recording.recording_id,
            )
        )
    candidate_channel_ids = tuple(channel["channel_id"] for channel in channels)
    is_mono_mix_candidate = (
        allow_mono_mix and len(recording.channels) > 1 and candidate_channel_ids == ("mono-mix",)
    )
    allowed_transform_chain_ids = (
        {f"{recording.transform_chain_id}-mono-mix"} if is_mono_mix_candidate else {recording.transform_chain_id}
    )
    if timeline["transform_chain_id"] not in allowed_transform_chain_ids:
        issues.append(
            _issue(
                "audio_metadata_mismatch",
                "candidate audio transform_chain_id does not match canonical reference",
                recording_id=recording.recording_id,
            )
        )
    expected_channel_count = len(recording.channels)
    if allow_mono_mix:
        allowed_channel_counts = {1, expected_channel_count}
    else:
        allowed_channel_counts = {expected_channel_count}
    if audio["channel_count"] not in allowed_channel_counts or len(channels) not in allowed_channel_counts:
        issues.append(
            _issue(
                "audio_metadata_mismatch",
                "candidate channel layout does not match expected fixture mode",
                recording_id=recording.recording_id,
            )
        )
    elif not _candidate_matches_expected_channels(
        channels,
        tuple(channel.channel_id for channel in recording.channels),
        allow_mono_mix=allow_mono_mix,
    ):
        issues.append(
            _issue(
                "audio_metadata_mismatch",
                "candidate channel ids do not match expected fixture mode",
                recording_id=recording.recording_id,
            )
        )
    return _result(issues)


def validate_scoring_exports(
    *,
    artifact_paths: dict[str, str],
    required_exports: tuple[str, ...] = ("rttm", "uem"),
) -> FixtureValidationResult:
    """Validate required scoring export artifact paths before benchmark execution."""

    artifact_paths = _validate_string_map(artifact_paths, "artifact_paths")
    checked_files: list[str] = []
    issues: list[FixtureValidationIssue] = []
    for export_name in required_exports:
        export_name = _require_id(export_name, "required_exports")
        path_text = artifact_paths.get(export_name)
        if path_text is None:
            issues.append(_issue("missing_scoring_export", f"missing scoring export path: {export_name}"))
            continue
        checked_files.append(path_text)
        if not Path(path_text).is_file():
            issues.append(
                _issue(
                    "missing_scoring_export",
                    f"missing scoring export file: {export_name}",
                    path=path_text,
                )
            )
            continue
        try:
            text = Path(path_text).read_text(encoding="utf-8")
            if export_name == "rttm":
                validate_rttm_text(text)
            elif export_name == "uem":
                validate_uem_text(text)
        except (OSError, ValidationError) as exc:
            issues.append(
                _issue(
                    "schema_validation",
                    f"invalid scoring export {export_name}: {exc}",
                    path=path_text,
                )
            )
    return _result(issues, checked_files=tuple(checked_files))


def validate_fixture_gate(
    *,
    manifest: DatasetManifest | None = None,
    expected_files_root: str | Path = ".",
    canonical_payloads: tuple[dict[str, Any], ...] = (),
    candidate_payloads: tuple[tuple[dict[str, Any], CanonicalRecording], ...] = (),
    scoring_policy: ScoringPolicyManifest | None = None,
    artifact_paths: dict[str, str] | None = None,
    audio_transform_caches: tuple[AudioTransformCacheCheck, ...] = (),
    minimum_slice_support: int = 1,
    allow_mono_mix: bool = False,
) -> FixtureValidationResult:
    """Run the full fixture gate and aggregate invalid_fixture issues."""

    results: list[FixtureValidationResult] = []
    if manifest is not None:
        results.append(validate_manifest_expected_files(manifest, root=expected_files_root))
    canonical_recordings: list[CanonicalRecording] = []
    canonical_issues: list[FixtureValidationIssue] = []
    canonical_recording_ids: set[str] = set()
    for payload in canonical_payloads:
        try:
            recording = recording_from_dict(payload)
        except ValidationError as exc:
            results.append(_result((_issue(_category_for_model_error(str(exc)), str(exc)),)))
            continue
        recording_issues = list(_canonical_recording_issues(recording, scoring_policy=scoring_policy))
        if recording.recording_id in canonical_recording_ids:
            recording_issues.append(
                _issue(
                    "schema_validation",
                    f"duplicate canonical recording_id: {recording.recording_id}",
                    recording_id=recording.recording_id,
                )
            )
        else:
            canonical_recording_ids.add(recording.recording_id)
            canonical_recordings.append(recording)
        canonical_issues.extend(recording_issues)
    if canonical_recordings:
        results.append(
            _result(
                canonical_issues,
                slice_metadata=build_fixture_slice_metadata(
                    tuple(canonical_recordings),
                    minimum_support=minimum_slice_support,
                ),
            )
        )
    for payload, recording in candidate_payloads:
        results.append(
            validate_candidate_bundle_against_reference(
                payload,
                recording,
                allow_mono_mix=allow_mono_mix,
            )
        )
    if artifact_paths is not None:
        results.append(validate_scoring_exports(artifact_paths=artifact_paths))
    for cache_check in _tuple_of(
        audio_transform_caches,
        AudioTransformCacheCheck,
        "audio_transform_caches",
    ):
        results.append(
            validate_audio_transform_cache(
                cache_check.manifest,
                canonical_audio_path=cache_check.canonical_audio_path,
                original_audio_path=cache_check.original_audio_path,
                expected_config=cache_check.expected_config,
            )
        )
    return merge_fixture_validation_results(*results)


def merge_fixture_validation_results(*results: FixtureValidationResult) -> FixtureValidationResult:
    issues: list[FixtureValidationIssue] = []
    slice_metadata: list[FixtureSliceMetadata] = []
    checked_files: list[str] = []
    for result in results:
        if not isinstance(result, FixtureValidationResult):
            raise ValidationError("results must be FixtureValidationResult values")
        issues.extend(result.issues)
        slice_metadata.extend(result.slice_metadata)
        checked_files.extend(result.checked_files)
    return _result(
        issues,
        slice_metadata=_merge_slice_metadata(tuple(slice_metadata)),
        checked_files=tuple(checked_files),
    )


def build_fixture_slice_metadata(
    recordings: tuple[CanonicalRecording, ...],
    *,
    minimum_support: int = 1,
) -> tuple[FixtureSliceMetadata, ...]:
    """Build report-ready validation slices for benchmark fixtures."""

    recordings = _tuple_of(recordings, CanonicalRecording, "recordings")
    minimum_support = _require_positive_int(minimum_support, "minimum_support")
    grouped: dict[tuple[str, str], list[tuple[CanonicalRecording, dict[str, Any]]]] = {}
    for recording in recordings:
        for dimension, value, metrics in _recording_slice_values(recording):
            grouped.setdefault((dimension, value), []).append((recording, metrics))

    result: list[FixtureSliceMetadata] = []
    for (dimension, value), entries in sorted(grouped.items()):
        recording_ids = tuple(recording.recording_id for recording, _ in entries)
        support_count = len(recording_ids)
        status: FixtureSliceStatus = "ready" if support_count >= minimum_support else "insufficient_support"
        result.append(
            FixtureSliceMetadata(
                slice_id=f"{dimension}:{value}",
                dimension=dimension,
                value=value,
                status=status,
                support_count=support_count,
                minimum_support=minimum_support,
                recording_ids=recording_ids,
                metrics=_merge_metrics(tuple(metrics for _, metrics in entries)),
            )
        )
    return tuple(result)


def _recording_slice_values(recording: CanonicalRecording) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    speech_ms = _speech_ms(recording)
    overlap_ms = _overlap_ms(recording)
    speech_ratio = _safe_ratio(speech_ms, recording.duration_ms)
    overlap_ratio = _safe_ratio(overlap_ms, speech_ms)
    return (
        ("speaker_count", str(len(recording.speakers)), {"speaker_count": len(recording.speakers)}),
        ("overlap_ratio", _ratio_bucket(overlap_ratio), {"overlap_ratio": round(overlap_ratio, 6)}),
        ("speech_ratio", _ratio_bucket(speech_ratio), {"speech_ratio": round(speech_ratio, 6)}),
        ("duration_bucket", _duration_bucket(recording.duration_ms), {"duration_ms": recording.duration_ms}),
        ("channel_mode", _channel_mode(recording), {"channel_count": len(recording.channels)}),
        (
            "known_count_mode",
            "known_speaker_count" if recording.speakers else "unknown_speaker_count",
            {"speaker_count": len(recording.speakers)},
        ),
    )


def _speech_ms(recording: CanonicalRecording) -> int:
    intervals = [(span.start_ms, span.end_ms) for span in recording.speaker_spans]
    if not intervals:
        intervals = [(word.start_ms, word.end_ms) for word in recording.words]
    return _interval_union_ms(tuple(intervals))


def _overlap_ms(recording: CanonicalRecording) -> int:
    intervals = [(span.start_ms, span.end_ms) for span in recording.speaker_spans if span.overlap]
    intervals.extend(_overlapping_speaker_span_intervals(recording.speaker_spans))
    if not intervals:
        intervals = [(word.start_ms, word.end_ms) for word in recording.words if word.overlap]
    return _interval_union_ms(tuple(intervals))


def _has_overlapping_speaker_spans(spans: tuple[Any, ...]) -> bool:
    return bool(_overlapping_speaker_span_intervals(spans))


def _overlapping_speaker_span_intervals(spans: tuple[Any, ...]) -> list[tuple[int, int]]:
    overlaps: list[tuple[int, int]] = []
    active: list[tuple[int, str]] = []
    for span in sorted(spans, key=lambda item: (item.start_ms, item.end_ms, item.speaker_ref)):
        active = [(end_ms, speaker_ref) for end_ms, speaker_ref in active if end_ms > span.start_ms]
        overlaps.extend(
            (span.start_ms, min(end_ms, span.end_ms)) for end_ms, speaker_ref in active if speaker_ref != span.speaker_ref
        )
        active.append((span.end_ms, span.speaker_ref))
    return overlaps


def _interval_union_ms(intervals: tuple[tuple[int, int], ...]) -> int:
    total = 0
    current_start: int | None = None
    current_end: int | None = None
    for start_ms, end_ms in sorted(intervals):
        if current_start is None or current_end is None:
            current_start = start_ms
            current_end = end_ms
            continue
        if start_ms <= current_end:
            current_end = max(current_end, end_ms)
            continue
        total += current_end - current_start
        current_start = start_ms
        current_end = end_ms
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, numerator / denominator))


def _ratio_bucket(value: float) -> str:
    if value == 0:
        return "none"
    if value < 0.10:
        return "low"
    if value < 0.35:
        return "medium"
    return "high"


def _duration_bucket(duration_ms: int) -> str:
    if duration_ms < 60_000:
        return "under_1m"
    if duration_ms < 10 * 60_000:
        return "1m_to_10m"
    return "over_10m"


def _channel_mode(recording: CanonicalRecording) -> str:
    if len(recording.channels) <= 1:
        return "mono"
    return "multichannel"


def _candidate_matches_expected_channels(
    channels: list[dict[str, Any]],
    expected_channel_ids: tuple[str, ...],
    *,
    allow_mono_mix: bool,
) -> bool:
    candidate_channel_ids = tuple(channel["channel_id"] for channel in channels)
    if candidate_channel_ids == expected_channel_ids:
        return True
    return allow_mono_mix and len(expected_channel_ids) > 1 and candidate_channel_ids == ("mono-mix",)


def _merge_slice_metadata(slices: tuple[FixtureSliceMetadata, ...]) -> tuple[FixtureSliceMetadata, ...]:
    grouped: dict[tuple[str, str, int], list[FixtureSliceMetadata]] = {}
    for item in slices:
        grouped.setdefault((item.dimension, item.value, item.minimum_support), []).append(item)

    result: list[FixtureSliceMetadata] = []
    for (dimension, value, minimum_support), items in sorted(grouped.items()):
        recording_ids = _unique_preserving_order(
            recording_id for item in items for recording_id in item.recording_ids
        )
        support_count = len(recording_ids)
        status: FixtureSliceStatus = "ready" if support_count >= minimum_support else "insufficient_support"
        result.append(
            FixtureSliceMetadata(
                slice_id=f"{dimension}:{value}",
                dimension=dimension,
                value=value,
                status=status,
                support_count=support_count,
                minimum_support=minimum_support,
                recording_ids=recording_ids,
                metrics=_merge_metrics(tuple(item.metrics for item in items)),
            )
        )
    return tuple(result)


def _merge_metrics(values: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        for key, item in value.items():
            if isinstance(item, (int, float)):
                merged[key] = max(merged.get(key, item), item)
            else:
                merged[key] = item
    return merged


def _category_for_model_error(message: str) -> FixtureIssueCategory:
    if (
        "start_ms must be >= 0" in message
        or "end_ms must be greater" in message
        or "ends after recording duration" in message
    ):
        return "invalid_interval"
    if "references unknown speaker_ref" in message:
        return "unresolved_speaker"
    return "schema_validation"


def _category_for_candidate_error(message: str) -> FixtureIssueCategory:
    if "forbidden in candidate bundles" in message:
        return "reference_leakage"
    return "schema_validation"


def _issue(
    category: FixtureIssueCategory,
    message: str,
    *,
    path: str | None = None,
    recording_id: str | None = None,
) -> FixtureValidationIssue:
    return FixtureValidationIssue(
        issue_id=f"{category}-{hashlib.sha256(message.encode('utf-8')).hexdigest()[:12]}",
        category=category,
        message=message,
        path=path,
        recording_id=recording_id,
    )


def _result(
    issues: tuple[FixtureValidationIssue, ...] | list[FixtureValidationIssue],
    *,
    slice_metadata: tuple[FixtureSliceMetadata, ...] = (),
    checked_files: tuple[str, ...] = (),
) -> FixtureValidationResult:
    issues = tuple(issues)
    return FixtureValidationResult(
        status="invalid_fixture" if issues else "valid",
        issues=issues,
        slice_metadata=slice_metadata,
        checked_files=checked_files,
    )


def _validate_audio_cache_file(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
    issues: list[FixtureValidationIssue],
) -> str:
    path = Path(path)
    path_text = path.as_posix()
    if not path.is_file():
        issues.append(_issue("missing_file", f"missing cached {label} file", path=path_text))
        return path_text
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        issues.append(_issue("checksum_mismatch", f"cached {label} sha256 mismatch", path=path_text))
    return path_text


def _validate_string_map(value: object, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[_require_id(key, f"{field_name}.key")] = _require_id(item, f"{field_name}.{key}")
    return result


def _validate_json_object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValidationError(f"{field_name} field names must be strings")
        result[key] = _validate_json_value(item, f"{field_name}.{key}")
    return result


def _validate_json_value(value: object, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{field_name} must be finite")
        return value
    if isinstance(value, list):
        return [_validate_json_value(item, f"{field_name}[]") for item in value]
    if isinstance(value, tuple):
        return tuple(_validate_json_value(item, f"{field_name}[]") for item in value)
    if isinstance(value, dict):
        return _validate_json_object(value, field_name)
    raise ValidationError(f"{field_name} must be JSON-compatible")


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    return value


def _tuple_of(values: object, item_type: type[Any], field_name: str) -> tuple[Any, ...]:
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{field_name} must be an iterable") from exc
    for index, item in enumerate(items):
        if not isinstance(item, item_type):
            raise ValidationError(f"{field_name}[{index}] must be a {item_type.__name__}")
    return items


def _unique_tuple_of_ids(values: object, field_name: str) -> tuple[str, ...]:
    result = tuple(_require_id(value, field_name) for value in _sequence(values, field_name))
    seen: set[str] = set()
    for value in result:
        if value in seen:
            raise ValidationError(f"{field_name} contains duplicate id: {value}")
        seen.add(value)
    return result


def _unique_tuple_of_paths(values: object, field_name: str) -> tuple[str, ...]:
    result = tuple(_require_text(value, field_name) for value in _sequence(values, field_name))
    seen: set[str] = set()
    unique: list[str] = []
    for value in result:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return tuple(unique)


def _unique_preserving_order(values: object) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        value = _require_id(value, "fixture_slice.recording_ids")
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return tuple(unique)


def _sequence(value: object, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValidationError(f"{field_name} must be an array")
    return tuple(value)


def _require_id(value: object, field_name: str) -> str:
    if value is None:
        raise ValidationError(f"{field_name} is required")
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValidationError(f"{field_name} is required")
    return value


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValidationError(f"{field_name} is required")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _optional_path_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return _require_text(value, field_name)


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if value < 0:
        raise ValidationError(f"{field_name} must be >= 0")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    value = _require_non_negative_int(value, field_name)
    if value <= 0:
        raise ValidationError(f"{field_name} must be greater than 0")
    return value
