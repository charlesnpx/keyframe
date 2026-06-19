"""AMI corpus normalization into canonical diarization references."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from keyframe.diarization.adapters import (
    BenchmarkArtifactLayout,
    DatasetCacheConfig,
    DatasetExportResult,
    DatasetPreparationPlan,
    DatasetValidationResult,
    plan_dataset_preparation,
)
from keyframe.diarization.bundles import ReferenceBundle, build_candidate_bundle
from keyframe.diarization.io import write_recording_json
from keyframe.diarization.manifests import (
    DatasetManifest,
    DatasetSplitManifest,
    read_dataset_manifest_json,
)
from keyframe.diarization.models import (
    CanonicalRecording,
    CanonicalWord,
    ChannelRecord,
    DisplayLabel,
    ScoringRegion,
    SpeakerRecord,
    SpeakerSpan,
    ValidationError,
)


AMI_SAMPLE_RATE_HZ = 16_000
_DEFAULT_MANIFEST_PATH = Path(__file__).parent / "dataset_manifests" / "ami.json"
_WORD_TAGS = frozenset({"w", "word"})
_SEGMENT_TAGS = frozenset({"segment", "seg", "turn", "speakerturn"})


@dataclass(frozen=True)
class AMIChannelMetadata:
    """Candidate-safe microphone/channel metadata plus reference-only source linkage."""

    channel_id: str
    name: str | None = None
    source_speaker_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_id", _require_id(self.channel_id, "ami_channel.channel_id"))
        object.__setattr__(self, "name", _optional_text(self.name, "ami_channel.name"))
        object.__setattr__(
            self,
            "source_speaker_id",
            _optional_id(self.source_speaker_id, "ami_channel.source_speaker_id"),
        )


@dataclass(frozen=True)
class AMIWordAnnotation:
    """One AMI NXT-style word annotation before canonical speaker remapping."""

    source_word_id: str
    source_speaker_id: str
    text: str
    start_ms: int
    end_ms: int
    channel_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_word_id", _require_id(self.source_word_id, "ami_word.source_word_id"))
        object.__setattr__(
            self,
            "source_speaker_id",
            _require_id(self.source_speaker_id, "ami_word.source_speaker_id"),
        )
        object.__setattr__(self, "text", _require_text(self.text, "ami_word.text"))
        object.__setattr__(self, "channel_id", _optional_id(self.channel_id, "ami_word.channel_id"))
        start_ms = _require_non_negative_int(self.start_ms, "ami_word.start_ms")
        end_ms = _require_non_negative_int(self.end_ms, "ami_word.end_ms")
        if end_ms <= start_ms:
            raise ValidationError("ami_word.end_ms must be greater than start_ms")
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)


@dataclass(frozen=True)
class AMISpeakerSegment:
    """One AMI transcript/speaker segment before canonical speaker remapping."""

    source_segment_id: str
    source_speaker_id: str
    start_ms: int
    end_ms: int
    channel_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_segment_id",
            _require_id(self.source_segment_id, "ami_segment.source_segment_id"),
        )
        object.__setattr__(
            self,
            "source_speaker_id",
            _require_id(self.source_speaker_id, "ami_segment.source_speaker_id"),
        )
        object.__setattr__(self, "channel_id", _optional_id(self.channel_id, "ami_segment.channel_id"))
        start_ms = _require_non_negative_int(self.start_ms, "ami_segment.start_ms")
        end_ms = _require_non_negative_int(self.end_ms, "ami_segment.end_ms")
        if end_ms <= start_ms:
            raise ValidationError("ami_segment.end_ms must be greater than start_ms")
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)


@dataclass(frozen=True)
class AMIRecordingSource:
    """Local AMI source files for one recording."""

    recording_id: str
    word_paths: tuple[str | Path, ...]
    segment_paths: tuple[str | Path, ...] = ()
    channels_path: str | Path | None = None
    duration_ms: int | None = None
    sample_rate_hz: int = AMI_SAMPLE_RATE_HZ
    original_audio_id: str | None = None
    canonical_audio_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "recording_id", _require_id(self.recording_id, "ami_recording.recording_id"))
        word_paths = tuple(Path(path) for path in self.word_paths)
        if not word_paths:
            raise ValidationError("ami_recording.word_paths is required")
        object.__setattr__(self, "word_paths", word_paths)
        object.__setattr__(self, "segment_paths", tuple(Path(path) for path in self.segment_paths))
        object.__setattr__(self, "channels_path", None if self.channels_path is None else Path(self.channels_path))
        if self.duration_ms is not None:
            object.__setattr__(
                self,
                "duration_ms",
                _require_positive_int(self.duration_ms, "ami_recording.duration_ms"),
            )
        object.__setattr__(
            self,
            "sample_rate_hz",
            _require_positive_int(self.sample_rate_hz, "ami_recording.sample_rate_hz"),
        )
        object.__setattr__(
            self,
            "original_audio_id",
            _optional_id(self.original_audio_id, "ami_recording.original_audio_id"),
        )
        object.__setattr__(
            self,
            "canonical_audio_id",
            _optional_id(self.canonical_audio_id, "ami_recording.canonical_audio_id"),
        )


class AMIAdapter:
    """Dataset adapter for a local AMI cache or AMI-shaped fixture."""

    adapter_id = "ami"

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        source_root: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(_DEFAULT_MANIFEST_PATH if manifest_path is None else manifest_path)
        self.manifest = read_dataset_manifest_json(self.manifest_path)
        self.source_root = None if source_root is None else Path(source_root)

    def describe_splits(self) -> tuple[DatasetSplitManifest, ...]:
        return self.manifest.splits

    def prepare(self, cache: DatasetCacheConfig, *, download: bool = False) -> DatasetPreparationPlan:
        return plan_dataset_preparation(self.manifest, cache, download=download)

    def validate_source(self, split_id: str, cache: DatasetCacheConfig) -> DatasetValidationResult:
        split = _split_by_id(self.manifest, split_id)
        checked_files: list[str] = []
        errors: list[str] = []

        for path in split.expected_file_paths:
            resolved = _resolve_manifest_path(self.manifest_path, path)
            checked_files.append(resolved.as_posix())
            if not resolved.is_file():
                errors.append(f"missing split descriptor: {path}")

        source_root = self._effective_source_root(cache)
        if source_root is None:
            errors.append("missing AMI source root; set adapter source_root or dataset_cache.cache_root")
        else:
            checked_files.append(source_root.as_posix())
            for recording_id in split.recording_ids:
                try:
                    source = _recording_source_from_root(recording_id, source_root)
                except ValidationError as exc:
                    errors.append(str(exc))
                    continue
                checked_files.extend(path.as_posix() for path in source.word_paths)
                checked_files.extend(path.as_posix() for path in source.segment_paths)
                if source.channels_path is not None:
                    checked_files.append(source.channels_path.as_posix())

        return DatasetValidationResult(
            dataset_id=self.manifest.dataset_id,
            split_id=split.split_id,
            valid=not errors,
            checked_files=tuple(checked_files),
            errors=tuple(errors),
        )

    def normalize(self, split_id: str, cache: DatasetCacheConfig) -> tuple[CanonicalRecording, ...]:
        split = _split_by_id(self.manifest, split_id)
        source_root = self._effective_source_root(cache)
        if source_root is None:
            raise ValidationError("AMI normalization requires adapter source_root or dataset_cache.cache_root")
        return tuple(
            normalize_ami_recording(_recording_source_from_root(recording_id, source_root))
            for recording_id in split.recording_ids
        )

    def export_reference(
        self,
        split_id: str,
        recordings: tuple[CanonicalRecording, ...],
        artifact_layout: BenchmarkArtifactLayout,
    ) -> tuple[DatasetExportResult, ...]:
        if not recordings:
            raise ValidationError("AMIAdapter.export_reference requires at least one recording")
        return tuple(self._export_one(split_id, recording, artifact_layout) for recording in recordings)

    def export_references(
        self,
        split_id: str,
        recordings: tuple[CanonicalRecording, ...],
        artifact_layout: BenchmarkArtifactLayout,
    ) -> tuple[DatasetExportResult, ...]:
        return self.export_reference(split_id, recordings, artifact_layout)

    def _effective_source_root(self, cache: DatasetCacheConfig) -> Path | None:
        if not isinstance(cache, DatasetCacheConfig):
            raise ValidationError("cache must be a DatasetCacheConfig")
        root = self.source_root or cache.cache_path
        if root is None:
            return None
        root = Path(root)
        nested_ami_root = root / "ami"
        return nested_ami_root if nested_ami_root.is_dir() else root

    def _export_one(
        self,
        split_id: str,
        recording: CanonicalRecording,
        artifact_layout: BenchmarkArtifactLayout,
    ) -> DatasetExportResult:
        _split_by_id(self.manifest, split_id)
        reference = build_ami_reference_bundle(recording, artifact_id=f"{split_id}-{recording.recording_id}-reference")
        artifact_paths = _write_ami_artifacts(split_id, recording, reference, artifact_layout)
        return DatasetExportResult(
            dataset_id=self.manifest.dataset_id,
            split_id=split_id,
            reference_bundle=reference,
            artifact_paths=artifact_paths,
        )


def read_ami_channels_xml(path: str | Path) -> tuple[AMIChannelMetadata, ...]:
    root = _parse_xml(path)
    channels: list[AMIChannelMetadata] = []
    for index, element in enumerate(root.iter(), start=1):
        tag = _local_name(element.tag)
        if tag not in {"channel", "microphone", "mic", "track"}:
            continue
        channel_id = _xml_attr(element, "channel_id", "id", "mic_id", "microphone_id") or f"ch-{index}"
        channels.append(
            AMIChannelMetadata(
                channel_id=channel_id,
                name=_xml_attr(element, "name", "label", "track_name"),
                source_speaker_id=_xml_attr(
                    element,
                    "source_speaker_id",
                    "speaker_id",
                    "participant_id",
                    "participant",
                    "agent",
                ),
            )
        )
    return tuple(channels)


def read_ami_words_xml(
    path: str | Path,
    *,
    source_speaker_id: str | None = None,
    channel_id: str | None = None,
) -> tuple[AMIWordAnnotation, ...]:
    path = Path(path)
    root = _parse_xml(path)
    fallback_speaker_id = source_speaker_id or _source_speaker_id_from_path(path, marker="words")
    words: list[AMIWordAnnotation] = []
    for index, element in enumerate(root.iter(), start=1):
        if _local_name(element.tag) not in _WORD_TAGS:
            continue
        text = _normalized_text(element)
        if not text:
            continue
        source_id = (
            _xml_attr(element, "source_speaker_id", "speaker_id", "speaker", "participant_id", "participant", "agent")
            or fallback_speaker_id
        )
        if source_id is None:
            raise ValidationError(f"AMI word file has no source speaker id: {path}")
        words.append(
            AMIWordAnnotation(
                source_word_id=_xml_attr(element, "word_id", "id") or f"{path.stem}-{index}",
                source_speaker_id=source_id,
                text=text,
                start_ms=_time_to_ms(_required_xml_attr(element, path, "starttime", "start", "start_time")),
                end_ms=_time_to_ms(_required_xml_attr(element, path, "endtime", "end", "end_time")),
                channel_id=_xml_attr(element, "channel_id", "channel", "mic", "microphone") or channel_id,
            )
        )
    return tuple(words)


def read_ami_segments_xml(
    path: str | Path,
    *,
    source_speaker_id: str | None = None,
    channel_id: str | None = None,
) -> tuple[AMISpeakerSegment, ...]:
    path = Path(path)
    root = _parse_xml(path)
    fallback_speaker_id = source_speaker_id or _source_speaker_id_from_path(path, marker="segments")
    segments: list[AMISpeakerSegment] = []
    for index, element in enumerate(root.iter(), start=1):
        if _local_name(element.tag) not in _SEGMENT_TAGS:
            continue
        if _xml_attr(element, "starttime", "start", "start_time") is None:
            continue
        source_id = (
            _xml_attr(element, "source_speaker_id", "speaker_id", "speaker", "participant_id", "participant", "agent")
            or fallback_speaker_id
        )
        if source_id is None:
            raise ValidationError(f"AMI segment file has no source speaker id: {path}")
        segments.append(
            AMISpeakerSegment(
                source_segment_id=_xml_attr(element, "segment_id", "id") or f"{path.stem}-{index}",
                source_speaker_id=source_id,
                start_ms=_time_to_ms(_required_xml_attr(element, path, "starttime", "start", "start_time")),
                end_ms=_time_to_ms(_required_xml_attr(element, path, "endtime", "end", "end_time")),
                channel_id=_xml_attr(element, "channel_id", "channel", "mic", "microphone") or channel_id,
            )
        )
    return tuple(segments)


def normalize_ami_recording(source: AMIRecordingSource) -> CanonicalRecording:
    if not isinstance(source, AMIRecordingSource):
        raise ValidationError("source must be an AMIRecordingSource")
    channels = _read_channels(source.channels_path)
    channel_by_source_speaker = {
        channel.source_speaker_id: channel.channel_id for channel in channels if channel.source_speaker_id is not None
    }

    source_words: list[AMIWordAnnotation] = []
    for path in sorted(source.word_paths):
        speaker_id = _source_speaker_id_from_path(path, marker="words")
        source_words.extend(
            _with_inferred_word_channels(
                read_ami_words_xml(path, source_speaker_id=speaker_id),
                channel_by_source_speaker,
            )
        )
    if not source_words:
        raise ValidationError(f"AMI recording has no word annotations: {source.recording_id}")

    source_segments: list[AMISpeakerSegment] = []
    for path in sorted(source.segment_paths):
        speaker_id = _source_speaker_id_from_path(path, marker="segments")
        source_segments.extend(
            _with_inferred_segment_channels(
                read_ami_segments_xml(path, source_speaker_id=speaker_id),
                channel_by_source_speaker,
            )
        )
    if not source_segments:
        source_segments = list(_derive_segments_from_words(tuple(source_words)))

    channels = _complete_channels(channels, source_words, source_segments)
    speaker_order = _ordered_source_speakers(source_words, source_segments)
    speaker_ref_by_source = {
        source_speaker_id: f"spk-{index}" for index, source_speaker_id in enumerate(speaker_order, 1)
    }
    duration_ms = _recording_duration_ms(source, source_words, source_segments)
    canonical_spans = _canonical_spans(source_segments, speaker_ref_by_source)
    canonical_words = _canonical_words(source_words, speaker_ref_by_source, canonical_spans)

    return CanonicalRecording(
        recording_id=source.recording_id,
        original_audio_id=source.original_audio_id or f"ami-{source.recording_id}-original",
        canonical_audio_id=source.canonical_audio_id or f"ami-{source.recording_id}-canonical",
        timeline_id=f"ami-{source.recording_id}-timeline",
        duration_ms=duration_ms,
        transform_chain_id="identity",
        sample_rate_hz=source.sample_rate_hz,
        channels=tuple(ChannelRecord(channel_id=channel.channel_id, name=channel.name) for channel in channels),
        speakers=tuple(
            SpeakerRecord(
                speaker_ref=speaker_ref_by_source[source_speaker_id],
                display_label=DisplayLabel(
                    label=f"person_{index}",
                    source="channel_metadata",
                    scope="recording",
                    confidence=1.0,
                    source_ref=source_speaker_id,
                ),
            )
            for index, source_speaker_id in enumerate(speaker_order, start=1)
        ),
        words=canonical_words,
        speaker_spans=canonical_spans,
        scoring_regions=_scoring_regions(channels, canonical_words, canonical_spans, duration_ms),
    )


def build_ami_reference_bundle(
    recording: CanonicalRecording,
    *,
    artifact_id: str | None = None,
    local_audio_sha256: str | None = None,
) -> ReferenceBundle:
    if not isinstance(recording, CanonicalRecording):
        raise ValidationError("recording must be a CanonicalRecording")
    evaluator_speaker_map = {
        speaker.speaker_ref: (
            speaker.display_label.source_ref
            if speaker.display_label is not None and speaker.display_label.source_ref is not None
            else speaker.speaker_ref
        )
        for speaker in recording.speakers
    }
    return ReferenceBundle.from_recording(
        recording,
        artifact_id=artifact_id or f"{recording.recording_id}-ami-reference",
        evaluator_speaker_map=evaluator_speaker_map,
        local_audio_sha256=local_audio_sha256,
        oracle_metadata={
            "dataset_id": "ami",
            "recording_id": recording.recording_id,
            "source_speaker_ids": list(evaluator_speaker_map.values()),
            "speaker_map": evaluator_speaker_map,
        },
    )


def _write_ami_artifacts(
    split_id: str,
    recording: CanonicalRecording,
    reference: ReferenceBundle,
    artifact_layout: BenchmarkArtifactLayout,
) -> dict[str, str]:
    canonical_path = Path(artifact_layout.canonical_references_dir) / f"{recording.recording_id}.canonical.json"
    reference_path = Path(artifact_layout.canonical_references_dir) / f"{recording.recording_id}.reference.json"
    product_candidate_path = (
        Path(artifact_layout.candidate_bundles_dir) / f"{recording.recording_id}.product_realistic.json"
    )
    authenticated_candidate_path = (
        Path(artifact_layout.candidate_bundles_dir) / f"{recording.recording_id}.authenticated_track_metadata.json"
    )

    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    write_recording_json(canonical_path, recording)
    _write_json(reference_path, reference.to_evaluator_dict())
    _write_json(
        product_candidate_path,
        build_candidate_bundle(
            reference,
            bundle_id=f"{split_id}-{recording.recording_id}-product-realistic",
            mode="product_realistic",
        ).to_dict(),
    )
    _write_json(
        authenticated_candidate_path,
        build_candidate_bundle(
            reference,
            bundle_id=f"{split_id}-{recording.recording_id}-authenticated-track-metadata",
            mode="authenticated_track_metadata",
        ).to_dict(),
    )
    return {
        "authenticated_track_metadata_candidate_bundle": authenticated_candidate_path.as_posix(),
        "canonical_reference": canonical_path.as_posix(),
        "candidate_bundle": product_candidate_path.as_posix(),
        "reference_bundle": reference_path.as_posix(),
    }


def _recording_source_from_root(recording_id: str, source_root: Path) -> AMIRecordingSource:
    source_root = Path(source_root)
    recording_dir = source_root / recording_id
    search_root = recording_dir if recording_dir.is_dir() else source_root
    word_paths = _recording_xml_paths(search_root, recording_id, "words")
    if not word_paths:
        raise ValidationError(f"missing AMI word annotations for recording: {recording_id}")
    return AMIRecordingSource(
        recording_id=recording_id,
        word_paths=word_paths,
        segment_paths=_recording_xml_paths(search_root, recording_id, "segments"),
        channels_path=_first_existing_xml(search_root, recording_id, "channels"),
    )


def _recording_xml_paths(search_root: Path, recording_id: str, marker: str) -> tuple[Path, ...]:
    marker_dir = search_root / marker
    paths: list[Path] = []
    if marker_dir.is_dir():
        pattern = "*.xml" if search_root.name == recording_id else f"{recording_id}*.xml"
        paths = sorted(marker_dir.glob(pattern))
    if not paths:
        paths = sorted(search_root.glob(f"{recording_id}*.{marker}.xml"))
    if not paths:
        paths = sorted(search_root.glob(f"**/{recording_id}*.{marker}.xml"))
    return tuple(path for path in paths if path.is_file())


def _first_existing_xml(search_root: Path, recording_id: str, marker: str) -> Path | None:
    candidates = (
        search_root / f"{marker}.xml",
        search_root / f"{recording_id}.{marker}.xml",
        search_root / marker / f"{recording_id}.{marker}.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(search_root.glob(f"**/{recording_id}*.{marker}.xml"))
    return matches[0] if matches else None


def _read_channels(path: Path | None) -> tuple[AMIChannelMetadata, ...]:
    if path is None:
        return ()
    return read_ami_channels_xml(path)


def _with_inferred_word_channels(
    words: tuple[AMIWordAnnotation, ...],
    channel_by_source_speaker: dict[str, str],
) -> tuple[AMIWordAnnotation, ...]:
    return tuple(
        word
        if word.channel_id is not None
        else AMIWordAnnotation(
            source_word_id=word.source_word_id,
            source_speaker_id=word.source_speaker_id,
            text=word.text,
            start_ms=word.start_ms,
            end_ms=word.end_ms,
            channel_id=channel_by_source_speaker.get(word.source_speaker_id),
        )
        for word in words
    )


def _with_inferred_segment_channels(
    segments: tuple[AMISpeakerSegment, ...],
    channel_by_source_speaker: dict[str, str],
) -> tuple[AMISpeakerSegment, ...]:
    return tuple(
        segment
        if segment.channel_id is not None
        else AMISpeakerSegment(
            source_segment_id=segment.source_segment_id,
            source_speaker_id=segment.source_speaker_id,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            channel_id=channel_by_source_speaker.get(segment.source_speaker_id),
        )
        for segment in segments
    )


def _derive_segments_from_words(
    words: tuple[AMIWordAnnotation, ...],
    *,
    max_gap_ms: int = 500,
) -> tuple[AMISpeakerSegment, ...]:
    segments: list[AMISpeakerSegment] = []
    grouped: dict[tuple[str, str | None], list[AMIWordAnnotation]] = {}
    for word in words:
        grouped.setdefault((word.source_speaker_id, word.channel_id), []).append(word)
    for (source_speaker_id, channel_id), group in sorted(grouped.items()):
        current_start: int | None = None
        current_end: int | None = None
        for word in sorted(group, key=lambda item: (item.start_ms, item.end_ms, item.source_word_id)):
            if current_start is None or current_end is None:
                current_start = word.start_ms
                current_end = word.end_ms
                continue
            if word.start_ms <= current_end + max_gap_ms:
                current_end = max(current_end, word.end_ms)
                continue
            segments.append(
                AMISpeakerSegment(
                    source_segment_id=f"{source_speaker_id}-{len(segments) + 1}",
                    source_speaker_id=source_speaker_id,
                    start_ms=current_start,
                    end_ms=current_end,
                    channel_id=channel_id,
                )
            )
            current_start = word.start_ms
            current_end = word.end_ms
        if current_start is not None and current_end is not None:
            segments.append(
                AMISpeakerSegment(
                    source_segment_id=f"{source_speaker_id}-{len(segments) + 1}",
                    source_speaker_id=source_speaker_id,
                    start_ms=current_start,
                    end_ms=current_end,
                    channel_id=channel_id,
                )
            )
    return tuple(segments)


def _complete_channels(
    channels: tuple[AMIChannelMetadata, ...],
    words: list[AMIWordAnnotation],
    segments: list[AMISpeakerSegment],
) -> tuple[AMIChannelMetadata, ...]:
    by_id = {channel.channel_id: channel for channel in channels}
    for channel_id in tuple(word.channel_id for word in words) + tuple(segment.channel_id for segment in segments):
        if channel_id is not None and channel_id not in by_id:
            by_id[channel_id] = AMIChannelMetadata(channel_id=channel_id)
    if not by_id:
        by_id["mix"] = AMIChannelMetadata(channel_id="mix", name="mixed")
    return tuple(by_id[channel_id] for channel_id in sorted(by_id))


def _ordered_source_speakers(
    words: list[AMIWordAnnotation],
    segments: list[AMISpeakerSegment],
) -> tuple[str, ...]:
    events = [(word.start_ms, index, word.source_speaker_id) for index, word in enumerate(words)]
    offset = len(events)
    events.extend(
        (segment.start_ms, offset + index, segment.source_speaker_id) for index, segment in enumerate(segments)
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for _, _, source_speaker_id in sorted(events):
        if source_speaker_id not in seen:
            seen.add(source_speaker_id)
            ordered.append(source_speaker_id)
    return tuple(ordered)


def _recording_duration_ms(
    source: AMIRecordingSource,
    words: list[AMIWordAnnotation],
    segments: list[AMISpeakerSegment],
) -> int:
    max_end_ms = max(
        tuple(word.end_ms for word in words) + tuple(segment.end_ms for segment in segments),
        default=0,
    )
    duration_ms = max(source.duration_ms or 0, max_end_ms)
    if duration_ms <= 0:
        raise ValidationError(f"AMI recording has no positive duration: {source.recording_id}")
    return duration_ms


def _canonical_spans(
    segments: list[AMISpeakerSegment],
    speaker_ref_by_source: dict[str, str],
) -> tuple[SpeakerSpan, ...]:
    ordered_segments = sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.source_segment_id))
    overlap_flags = _overlapping_segment_indices(ordered_segments)
    return tuple(
        SpeakerSpan(
            span_id=f"span-{index}",
            speaker_ref=speaker_ref_by_source[segment.source_speaker_id],
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            channel_id=segment.channel_id,
            confidence=1.0,
            overlap=index - 1 in overlap_flags,
        )
        for index, segment in enumerate(ordered_segments, start=1)
    )


def _canonical_words(
    words: list[AMIWordAnnotation],
    speaker_ref_by_source: dict[str, str],
    spans: tuple[SpeakerSpan, ...],
) -> tuple[CanonicalWord, ...]:
    ordered_words = sorted(words, key=lambda item: (item.start_ms, item.end_ms, item.source_word_id))
    return tuple(
        CanonicalWord(
            word_id=f"w-{index}",
            text=word.text,
            start_ms=word.start_ms,
            end_ms=word.end_ms,
            speaker_ref=speaker_ref_by_source[word.source_speaker_id],
            channel_id=word.channel_id,
            speaker_confidence=1.0,
            overlap=_word_overlaps_different_speaker(word, spans, speaker_ref_by_source),
        )
        for index, word in enumerate(ordered_words, start=1)
    )


def _overlapping_segment_indices(segments: list[AMISpeakerSegment]) -> set[int]:
    overlap_indices: set[int] = set()
    for left_index, left in enumerate(segments):
        for right_index, right in enumerate(segments[left_index + 1 :], start=left_index + 1):
            if left.source_speaker_id == right.source_speaker_id:
                continue
            if _overlap_ms(left.start_ms, left.end_ms, right.start_ms, right.end_ms) > 0:
                overlap_indices.add(left_index)
                overlap_indices.add(right_index)
    return overlap_indices


def _word_overlaps_different_speaker(
    word: AMIWordAnnotation,
    spans: tuple[SpeakerSpan, ...],
    speaker_ref_by_source: dict[str, str],
) -> bool:
    speaker_ref = speaker_ref_by_source[word.source_speaker_id]
    for span in spans:
        if span.speaker_ref == speaker_ref:
            continue
        if _overlap_ms(word.start_ms, word.end_ms, span.start_ms, span.end_ms) > 0:
            return True
    return False


def _overlap_ms(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _scoring_regions(
    channels: tuple[AMIChannelMetadata, ...],
    words: tuple[CanonicalWord, ...],
    spans: tuple[SpeakerSpan, ...],
    duration_ms: int,
) -> tuple[ScoringRegion, ...]:
    regions: list[ScoringRegion] = []
    for index, channel in enumerate(channels, start=1):
        intervals = [
            (word.start_ms, word.end_ms)
            for word in words
            if word.channel_id == channel.channel_id or word.channel_id is None
        ]
        intervals.extend(
            (span.start_ms, span.end_ms)
            for span in spans
            if span.channel_id == channel.channel_id or span.channel_id is None
        )
        start_ms = min((start for start, _ in intervals), default=0)
        end_ms = max((end for _, end in intervals), default=duration_ms)
        regions.append(
            ScoringRegion(
                region_id=f"uem-{index}",
                start_ms=start_ms,
                end_ms=max(end_ms, start_ms + 1),
                channel_id=channel.channel_id,
            )
        )
    return tuple(regions)


def _split_by_id(manifest: DatasetManifest, split_id: str) -> DatasetSplitManifest:
    split_id = _require_id(split_id, "split_id")
    for split in manifest.splits:
        if split.split_id == split_id:
            return split
    raise ValidationError(f"unknown AMI split: {split_id}")


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    candidates = [Path.cwd() / raw_path, manifest_path.parent / raw_path.name]
    candidates.extend(parent / raw_path for parent in manifest_path.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _parse_xml(path: str | Path) -> ET.Element:
    path = Path(path)
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValidationError(f"AMI XML is invalid: {path}") from exc
    except OSError as exc:
        raise ValidationError(f"AMI XML cannot be read: {path}") from exc


def _local_name(value: str) -> str:
    if "}" in value:
        value = value.rsplit("}", 1)[1]
    if ":" in value:
        value = value.rsplit(":", 1)[1]
    return value.lower()


def _xml_attr(element: ET.Element, *names: str) -> str | None:
    wanted = {_local_name(name) for name in names}
    for key, value in element.attrib.items():
        if _local_name(key) in wanted:
            return value.strip() or None
    return None


def _required_xml_attr(element: ET.Element, path: Path, *names: str) -> str:
    value = _xml_attr(element, *names)
    if value is None:
        raise ValidationError(f"AMI XML element is missing {names[0]} in {path}")
    return value


def _normalized_text(element: ET.Element) -> str:
    return " ".join("".join(element.itertext()).split())


def _time_to_ms(value: str) -> int:
    value = value.strip()
    multiplier = 1 if value.endswith("ms") else 1000
    if value.endswith("ms"):
        value = value[:-2]
    try:
        seconds_or_ms = float(value)
    except ValueError as exc:
        raise ValidationError(f"AMI time value is invalid: {value}") from exc
    if not math.isfinite(seconds_or_ms) or seconds_or_ms < 0:
        raise ValidationError(f"AMI time value must be non-negative and finite: {value}")
    return round(seconds_or_ms * multiplier)


def _source_speaker_id_from_path(path: str | Path, *, marker: str) -> str | None:
    stem = Path(path).name
    suffix = f".{marker}.xml"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    parts = stem.split(".")
    if len(parts) >= 2:
        return parts[-1]
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_id(value: object, field_name: str) -> str:
    if value is None:
        raise ValidationError(f"{field_name} is required")
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValidationError(f"{field_name} is required")
    return value


def _optional_id(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_id(value, field_name)


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
