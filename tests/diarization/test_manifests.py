import hashlib
import json
from pathlib import Path

import pytest

from keyframe.diarization import (
    ALLOWED_DATASET_ACCESS_MODES,
    ALLOWED_DATASET_ROLES,
    DATASET_MANIFEST_SCHEMA_VERSION,
    DatasetAccess,
    DatasetManifest,
    DatasetSplitManifest,
    ExpectedDatasetFile,
    ScoringPolicyManifest,
    ValidationError,
    dataset_manifest_from_dict,
    dataset_manifest_json_dumps,
    dataset_manifest_json_loads,
    manifest_allows_default_ci_download,
    manifest_allows_default_full_download,
    read_dataset_manifest_json,
    write_dataset_manifest_json,
)


MANIFEST_DIR = Path("keyframe/diarization/dataset_manifests")


def _payload(name="ami.json"):
    return json.loads((MANIFEST_DIR / name).read_text(encoding="utf-8"))


def _valid_payload():
    return _payload()


def test_dataset_roles_and_access_modes_cover_planned_scope():
    assert ALLOWED_DATASET_ROLES == {
        "smoke_ci",
        "public_dev",
        "public_holdout",
        "gated_manual",
        "adversarial",
        "private_in_domain_acceptance",
    }
    assert ALLOWED_DATASET_ACCESS_MODES == {
        "public_direct",
        "public_manual",
        "auth_required",
        "local_only",
        "forbidden",
    }


def test_packaged_ami_manifest_loads_and_rewrites_byte_stable_json():
    path = MANIFEST_DIR / "ami.json"
    manifest = read_dataset_manifest_json(path)

    assert manifest.schema_version == DATASET_MANIFEST_SCHEMA_VERSION
    assert manifest.dataset_id == "ami"
    assert manifest.role == "public_dev"
    assert manifest.benchmarked is True
    assert manifest.access.mode == "public_manual"
    assert manifest.access.redistribution == "allowed"
    assert manifest.license_url == "https://groups.inf.ed.ac.uk/ami/corpus/license.shtml"
    assert manifest.attribution.startswith("AMI Meeting Corpus")
    assert [split.split_id for split in manifest.splits] == [
        "ami-smoke-ci",
        "ami-public-dev",
        "ami-public-holdout",
    ]
    assert dataset_manifest_json_dumps(manifest) == path.read_text(encoding="utf-8")


def test_expected_file_checksums_match_committed_manifest_files():
    manifest = read_dataset_manifest_json(MANIFEST_DIR / "ami.json")

    for expected_file in manifest.expected_files:
        file_path = Path(expected_file.path)
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()

        assert actual == expected_file.checksum_sha256
        assert file_path.stat().st_size == expected_file.size_bytes


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("license_url"), "license_url is required"),
        (lambda payload: payload.pop("attribution"), "attribution is required"),
        (lambda payload: payload.update({"expected_files": []}), "expected_files is required"),
        (lambda payload: payload.update({"splits": []}), "splits is required"),
        (lambda payload: payload.update({"scoring_policies": []}), "scoring_policies is required"),
        (
            lambda payload: payload["expected_files"][0].pop("checksum_sha256"),
            "expected_file.checksum_sha256 is required",
        ),
        (
            lambda payload: payload["expected_files"][0].update({"checksum_sha256": "A" * 64}),
            "expected_file.checksum must be a lowercase sha256 hex digest",
        ),
    ],
)
def test_benchmarked_manifest_requires_license_attribution_files_checksums_and_splits(mutate, message):
    payload = _valid_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        dataset_manifest_from_dict(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["splits"][0].update({"expected_file_paths": ["missing.json"]}),
            "references unknown file",
        ),
        (
            lambda payload: payload["splits"][0].update({"scoring_policy_id": "missing-policy"}),
            "scoring_policy_id is unknown",
        ),
        (
            lambda payload: payload["expected_files"][1].update({"path": payload["expected_files"][0]["path"]}),
            "duplicate expected_file.path",
        ),
    ],
)
def test_manifest_rejects_invalid_split_file_and_policy_references(mutate, message):
    payload = _valid_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        dataset_manifest_from_dict(payload)


def test_gated_and_local_placeholders_are_never_default_downloadable():
    for name in ("callhome_placeholder.json", "robustness_placeholder.json"):
        manifest = read_dataset_manifest_json(MANIFEST_DIR / name)

        assert manifest.benchmarked is False
        assert manifest.access.redistribution == "forbidden"
        assert manifest_allows_default_ci_download(manifest) is False
        assert manifest_allows_default_full_download(manifest) is False


def test_public_manual_ami_is_not_pulled_into_default_download_paths():
    manifest = read_dataset_manifest_json(MANIFEST_DIR / "ami.json")

    assert manifest.access.mode == "public_manual"
    assert manifest_allows_default_ci_download(manifest) is False
    assert manifest_allows_default_full_download(manifest) is False


def test_manifest_json_file_round_trip_is_stable(tmp_path):
    manifest = read_dataset_manifest_json(MANIFEST_DIR / "ami.json")
    target = tmp_path / "manifest.json"

    write_dataset_manifest_json(target, manifest)

    assert target.read_text(encoding="utf-8") == dataset_manifest_json_dumps(manifest)
    assert read_dataset_manifest_json(target).to_dict() == manifest.to_dict()
    assert dataset_manifest_json_loads(target.read_text(encoding="utf-8")).to_dict() == manifest.to_dict()


def test_manifest_dataclass_validation_accepts_smoke_ci_public_direct_case():
    manifest = DatasetManifest(
        dataset_id="smoke-fixture",
        name="Smoke Fixture",
        role="smoke_ci",
        access=DatasetAccess(mode="public_direct", redistribution="allowed", url="https://example.com/smoke.zip"),
        license_url="https://example.com/license",
        attribution="Synthetic smoke fixture",
        expected_files=(
            ExpectedDatasetFile(
                path="fixtures/smoke.json",
                checksum_sha256="0" * 64,
                file_role="manifest",
                size_bytes=0,
            ),
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
            ScoringPolicyManifest(
                policy_id="smoke-policy",
                version="1",
                description="Smoke policy",
            ),
        ),
    )

    assert manifest.default_ci_downloadable is True
    assert manifest.default_full_downloadable is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "split_id": "bad-split",
                "role": "smoke_ci",
                "expected_file_paths": "fixtures/smoke.json",
                "scoring_policy_id": "smoke-policy",
            },
            "sequence fields must be arrays",
        ),
        (
            {
                "split_id": "bad-split",
                "role": "smoke_ci",
                "expected_file_paths": ("fixtures/smoke.json",),
                "scoring_policy_id": "smoke-policy",
                "recording_ids": "recording-1",
            },
            "sequence fields must be arrays",
        ),
    ],
)
def test_public_split_constructor_rejects_string_sequence_fields(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        DatasetSplitManifest(**kwargs)


@pytest.mark.parametrize("path", ["/tmp/audio.wav", "../secret.wav", "C:/secret.wav", "..\\secret.wav"])
def test_manifest_paths_are_portable_relative_paths(path):
    with pytest.raises(ValidationError, match="relative path inside the dataset root|forward slashes"):
        ExpectedDatasetFile(path=path, checksum_sha256="0" * 64, file_role="audio")

    with pytest.raises(ValidationError, match="relative path inside the dataset root|forward slashes"):
        DatasetSplitManifest(
            split_id="bad-path",
            role="smoke_ci",
            expected_file_paths=(path,),
            scoring_policy_id="smoke-policy",
        )


def test_public_manifest_constructor_rejects_string_notes():
    with pytest.raises(ValidationError, match="sequence fields must be arrays"):
        DatasetManifest(
            dataset_id="smoke-fixture",
            name="Smoke Fixture",
            role="smoke_ci",
            access=DatasetAccess(mode="public_direct", redistribution="allowed", url="https://example.com/smoke.zip"),
            license_url="https://example.com/license",
            attribution="Synthetic smoke fixture",
            expected_files=(
                ExpectedDatasetFile(
                    path="fixtures/smoke.json",
                    checksum_sha256="0" * 64,
                    file_role="manifest",
                    size_bytes=0,
                ),
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
                ScoringPolicyManifest(
                    policy_id="smoke-policy",
                    version="1",
                    description="Smoke policy",
                ),
            ),
            notes="one note",
        )


def test_diarization_manifests_do_not_add_yaml_dependency():
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "yaml" not in pyproject_text.lower()
    assert dataset_manifest_json_loads.__module__ == "keyframe.diarization.manifests"
