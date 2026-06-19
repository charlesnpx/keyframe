import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from keyframe.diarization import (
    BENCHMARK_RUN_RECORD_SCHEMA_VERSION,
    BenchmarkRunRecord,
    CanonicalRecording,
    DatasetAccess,
    DatasetAdapter,
    DatasetCacheConfig,
    DatasetExportResult,
    DatasetManifest,
    DatasetPreparationPlan,
    DatasetSplitManifest,
    DatasetValidationResult,
    ExpectedDatasetFile,
    ReferenceBundle,
    ScoringPolicyManifest,
    ValidationError,
    benchmark_run_record_json_dumps,
    benchmark_run_record_json_loads,
    build_artifact_layout,
    build_candidate_bundle,
    create_benchmark_run_record,
    ensure_adapter_cache_policy,
    plan_dataset_preparation,
    read_benchmark_run_record_json,
    read_dataset_manifest_json,
    read_recording_json,
    write_benchmark_run_record_json,
)


MANIFEST_DIR = Path("keyframe/diarization/dataset_manifests")


@dataclass
class FakeDatasetAdapter:
    manifest_path: Path
    calls: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.adapter_id = "fake-adapter"
        self.manifest = read_dataset_manifest_json(self.manifest_path)

    def describe_splits(self):
        self.calls.append("describe_splits")
        return self.manifest.splits

    def prepare(self, cache, *, download=False):
        self.calls.append(f"prepare:{download}")
        return plan_dataset_preparation(self.manifest, cache, download=download)

    def validate_source(self, split_id, cache):
        self.calls.append(f"validate_source:{split_id}")
        return DatasetValidationResult(
            dataset_id=self.manifest.dataset_id,
            split_id=split_id,
            valid=True,
            checked_files=tuple(file.path for file in self.manifest.expected_files),
        )

    def normalize(self, split_id, cache):
        self.calls.append(f"normalize:{split_id}")
        return (read_recording_json("tests/diarization/fixtures/clean_two_speaker.json"),)

    def export_reference(self, split_id, recordings, artifact_layout):
        self.calls.append(f"export_reference:{split_id}")
        reference = ReferenceBundle.from_recording(recordings[0], artifact_id=f"{split_id}-reference")
        return DatasetExportResult(
            dataset_id=self.manifest.dataset_id,
            split_id=split_id,
            reference_bundle=reference,
            artifact_paths={"canonical_reference": f"{artifact_layout.canonical_references_dir}/{split_id}.json"},
        )


def _ami_manifest():
    return read_dataset_manifest_json(MANIFEST_DIR / "ami.json")


def test_fake_adapter_lifecycle_hooks_keep_scoring_centralized(tmp_path):
    adapter = FakeDatasetAdapter(MANIFEST_DIR / "ami.json")
    cache = DatasetCacheConfig(cache_root=str(tmp_path / "cache"))
    layout = build_artifact_layout(tmp_path / "artifacts")

    assert isinstance(adapter, DatasetAdapter)
    assert [split.split_id for split in adapter.describe_splits()] == ["ami-public-dev", "ami-public-holdout"]

    plan = adapter.prepare(cache)
    validation = adapter.validate_source("ami-public-dev", cache)
    recordings = adapter.normalize("ami-public-dev", cache)
    export = adapter.export_reference("ami-public-dev", recordings, layout)

    assert isinstance(plan, DatasetPreparationPlan)
    assert plan.network_required is False
    assert validation.valid is True
    assert isinstance(recordings[0], CanonicalRecording)
    assert export.dataset_id == "ami"
    assert export.reference_bundle.recording.recording_id == "fixture-clean-two-speaker"
    candidate = build_candidate_bundle(export.reference_bundle, bundle_id="candidate-fixture")
    assert candidate.product_quality_reportable is True
    assert "score" not in adapter.calls


def test_default_preparation_is_no_network_and_no_download(tmp_path):
    manifest = _ami_manifest()
    cache = DatasetCacheConfig(cache_root=str(tmp_path / "cache"))

    plan = plan_dataset_preparation(manifest, cache)

    assert plan.to_dict() == {
        "actions": ["validate_manifest", "validate_cache"],
        "cache_root": str(tmp_path / "cache"),
        "dataset_id": "ami",
        "download_required": False,
        "network_required": False,
        "split_ids": ["ami-public-dev", "ami-public-holdout"],
    }


def test_default_preparation_rejects_network_and_download_policy(tmp_path):
    manifest = _ami_manifest()

    with pytest.raises(ValidationError, match="allow_network is required"):
        DatasetCacheConfig(cache_root=str(tmp_path / "cache"), allow_download=True)

    cache = DatasetCacheConfig(cache_root=str(tmp_path / "cache"), allow_download=True, allow_network=True)
    with pytest.raises(ValidationError, match="must not allow network or downloads"):
        ensure_adapter_cache_policy(manifest, cache)

    no_download_cache = DatasetCacheConfig(cache_root=str(tmp_path / "cache"))
    with pytest.raises(ValidationError, match="downloads require an explicit allow_download"):
        plan_dataset_preparation(manifest, no_download_cache, download=True)

    download_cache = DatasetCacheConfig(
        cache_root=str(tmp_path / "cache"),
        allow_download=True,
        allow_network=True,
    )
    with pytest.raises(ValidationError, match="downloads require public_direct access"):
        plan_dataset_preparation(manifest, download_cache, download=True)


def test_explicit_download_plan_requires_public_direct_manifest_and_policy(tmp_path):
    manifest = DatasetManifest(
        dataset_id="smoke-fixture",
        name="Smoke Fixture",
        role="smoke_ci",
        access=DatasetAccess(mode="public_direct", redistribution="allowed", url="https://example.com/smoke.zip"),
        license_url="https://example.com/license",
        attribution="Synthetic smoke fixture",
        expected_files=(
            ExpectedDatasetFile(path="fixtures/smoke.json", checksum_sha256="0" * 64, file_role="manifest"),
        ),
        splits=(
            DatasetSplitManifest(
                split_id="smoke",
                role="smoke_ci",
                expected_file_paths=("fixtures/smoke.json",),
                scoring_policy_id="smoke-policy",
            ),
        ),
        scoring_policies=(
            ScoringPolicyManifest(policy_id="smoke-policy", version="1", description="Smoke policy"),
        ),
    )
    cache = DatasetCacheConfig(
        cache_root=str(tmp_path / "cache"),
        allow_download=True,
        allow_network=True,
    )

    plan = plan_dataset_preparation(manifest, cache, download=True)

    assert plan.network_required is True
    assert plan.download_required is True
    assert plan.actions == ("validate_manifest", "validate_cache", "download")


def test_full_benchmark_mode_requires_cache_for_non_redistributable_data(tmp_path):
    manifest = read_dataset_manifest_json(MANIFEST_DIR / "callhome_placeholder.json")

    with pytest.raises(ValidationError, match="require an explicit local cache path"):
        ensure_adapter_cache_policy(
            manifest,
            DatasetCacheConfig(),
            execution_mode="full_benchmark",
        )

    ensure_adapter_cache_policy(
        manifest,
        DatasetCacheConfig(cache_root=str(tmp_path / "licensed-cache")),
        execution_mode="full_benchmark",
    )


def test_benchmark_run_record_captures_snapshot_layout_cache_and_splits(tmp_path):
    manifest = _ami_manifest()
    cache = DatasetCacheConfig(cache_root=str(tmp_path / "cache"))

    record = create_benchmark_run_record(
        run_id="run-001",
        manifest=manifest,
        split_id="ami-public-dev",
        branch="feature/diarization-benchmark-platform",
        artifact_root=tmp_path / "artifacts",
        cache=cache,
        tuned_split_ids=("ami-public-dev",),
        evaluated_split_ids=("ami-public-dev", "ami-public-holdout"),
        derived_artifacts={
            "canonical_reference": "references/canonical/ami-public-dev.json",
            "candidate_bundle": "candidates/ami-public-dev.json",
            "report": "reports/ami-public-dev.json",
            "rttm": "exports/rttm/ami-public-dev.rttm",
            "uem": "exports/uem/ami-public-dev.uem",
            "raw_provider_json": "raw/provider_json/ami-public-dev.json",
            "run_record": "run_records/run-001.json",
        },
    )

    payload = record.to_dict()
    assert payload["schema_version"] == BENCHMARK_RUN_RECORD_SCHEMA_VERSION
    assert payload["dataset_snapshot"]["dataset_id"] == "ami"
    assert payload["split_id"] == "ami-public-dev"
    assert payload["cache_root"] == str(tmp_path / "cache")
    assert payload["artifact_layout"]["canonical_references_dir"].endswith("references/canonical")
    assert payload["artifact_layout"]["raw_provider_json_dir"].endswith("raw/provider_json")
    assert payload["tuned_split_ids"] == ["ami-public-dev"]
    assert payload["evaluated_split_ids"] == ["ami-public-dev", "ami-public-holdout"]
    assert payload["execution_mode"] == "default_no_network"
    assert payload["no_network"] is True


def test_benchmark_run_record_json_round_trip_is_stable(tmp_path):
    manifest = _ami_manifest()
    record = create_benchmark_run_record(
        run_id="run-001",
        manifest=manifest,
        split_id="ami-public-dev",
        branch="feature/diarization-benchmark-platform",
        artifact_root=tmp_path / "artifacts",
        cache=DatasetCacheConfig(cache_root=str(tmp_path / "cache")),
    )
    target = tmp_path / "run-record.json"

    write_benchmark_run_record_json(target, record)

    assert target.read_text(encoding="utf-8") == benchmark_run_record_json_dumps(record)
    assert read_benchmark_run_record_json(target).to_dict() == record.to_dict()
    assert benchmark_run_record_json_loads(target.read_text(encoding="utf-8")).to_dict() == record.to_dict()


def test_run_record_validation_rejects_unknown_split_and_inconsistent_evaluated_splits(tmp_path):
    manifest = _ami_manifest()

    with pytest.raises(ValidationError, match="run_record.split_id is unknown"):
        create_benchmark_run_record(
            run_id="run-001",
            manifest=manifest,
            split_id="missing-split",
            branch="feature/diarization-benchmark-platform",
            artifact_root=tmp_path / "artifacts",
            cache=DatasetCacheConfig(cache_root=str(tmp_path / "cache")),
        )

    with pytest.raises(ValidationError, match="must be included in evaluated_split_ids"):
        BenchmarkRunRecord(
            run_id="run-001",
            dataset_id="ami",
            dataset_snapshot=manifest.to_dict(),
            split_id="ami-public-dev",
            branch="feature/diarization-benchmark-platform",
            artifact_layout=build_artifact_layout(tmp_path / "artifacts"),
            cache_root=str(tmp_path / "cache"),
            tuned_split_ids=(),
            evaluated_split_ids=("ami-public-holdout",),
            execution_mode="default_no_network",
            no_network=True,
        )


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("run_record.tuned_split_ids", {"tuned_split_ids": ("missing-split",)}),
        ("run_record.evaluated_split_ids", {"evaluated_split_ids": ("missing-split",)}),
    ],
)
def test_run_record_creation_rejects_split_lists_outside_manifest(tmp_path, field_name, kwargs):
    manifest = _ami_manifest()

    with pytest.raises(ValidationError, match=f"{field_name} contains unknown split"):
        create_benchmark_run_record(
            run_id="run-001",
            manifest=manifest,
            split_id="ami-public-dev",
            branch="feature/diarization-benchmark-platform",
            artifact_root=tmp_path / "artifacts",
            cache=DatasetCacheConfig(cache_root=str(tmp_path / "cache")),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update({"dataset_id": "different-dataset"}),
            "dataset_id must match dataset_snapshot.dataset_id",
        ),
        (
            lambda payload: payload.update({"evaluated_split_ids": ["ami-public-dev", "not-in-manifest"]}),
            "run_record.evaluated_split_ids contains unknown split",
        ),
        (
            lambda payload: payload.update({"tuned_split_ids": ["not-in-manifest"]}),
            "run_record.tuned_split_ids contains unknown split",
        ),
    ],
)
def test_run_record_loader_rejects_snapshot_and_split_tampering(tmp_path, mutate, message):
    manifest = _ami_manifest()
    record = create_benchmark_run_record(
        run_id="run-001",
        manifest=manifest,
        split_id="ami-public-dev",
        branch="feature/diarization-benchmark-platform",
        artifact_root=tmp_path / "artifacts",
        cache=DatasetCacheConfig(cache_root=str(tmp_path / "cache")),
    )
    payload = record.to_dict()
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        benchmark_run_record_json_loads(json.dumps(payload))
