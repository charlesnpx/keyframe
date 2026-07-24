from __future__ import annotations

import binascii
import json
import shutil
import struct
import zlib
from pathlib import Path

from keyframe import release_evidence


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "release-frame-fixture"
TARGET_TEXT = (
    "SOURCE FIELDS Linked Risks Risk ID Description Dependencies",
    "PRIORITY FORM Risk Justification Other Considerations Override Justification",
    "CONSEQUENCE OF FAILURE Dependencies Comments Summary of Changes",
    "PAGE 2 OF 2 Current State Description Prepared By",
    "Signed on Behalf of the IMC By Approved Date Status Draft",
    (
        "Signed on Behalf of the IMC By Alyssa Leon Approved Date "
        "29APR2026 Status Approved"
    ),
)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)
    )


def _write_png(path: Path, *, color: tuple[int, int, int]) -> None:
    width = 1280
    height = 720
    row = b"\x00" + bytes(color) * width
    raw = row * height
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )


def write_frame_evidence_bundle(
    bundle: Path,
    *,
    system: str,
    machine: str,
    source_kind: str = "git",
    source_value: str = "a" * 40,
    version: str = "0.6.3",
) -> dict:
    fixture = bundle / "fixture"
    frames = bundle / "artifacts" / "frames"
    configuration = bundle / "configuration"
    fixture.mkdir(parents=True)
    frames.mkdir(parents=True)
    configuration.mkdir(parents=True)
    for name in (
        "metadata.json",
        "release-frame-fixture.mp4",
        "LICENSE.txt",
        "source.json",
        "generate_fixture.py",
    ):
        shutil.copy2(FIXTURE_ROOT / name, fixture / name)
    contract = release_evidence.load_fixture_contract(fixture / "metadata.json")
    qa_path = configuration / "qa-targets.json"
    release_evidence._atomic_write_json(
        qa_path,
        release_evidence.qa_targets_from_fixture(contract.metadata),
    )

    manifest_rows = []
    caption_rows = []
    for index, (target, text) in enumerate(
        zip(contract.metadata["targets"], TARGET_TEXT, strict=True),
        start=1,
    ):
        timestamp = float(target["time_seconds"])
        name = f"frame_{int(timestamp * 30):06d}_{timestamp:.2f}s.png"
        _write_png(
            frames / name,
            color=(20 * index, 30 + 10 * index, 150 - 10 * index),
        )
        tokens = list(release_evidence.normalize_tokens(text))
        manifest_rows.append(
            {
                "filename": name,
                "timestamp": timestamp,
                "ocr_tokens": tokens,
                "caption": text,
            }
        )
        caption_rows.append(
            {
                "file": name,
                "timestamp": timestamp,
                "ocr_text": text,
                "ocr_tokens": tokens,
                "caption": text,
            }
        )
    release_evidence._atomic_write_json(
        frames / "manifest.json",
        {"schema_version": 1, "frames": manifest_rows},
    )
    release_evidence._atomic_write_json(
        frames / "captions.json",
        caption_rows,
    )
    trace = {
        "schema": 1,
        "records": [
            {
                "event": "exit",
                "stage": "models.provenance",
                "payload": {
                    "models": [
                        {
                            "role": "visual-embedding",
                            "model_id": (
                                "open_clip/ViT-B-32/laion2b_s34b_b79k"
                            ),
                            "repository_revision": "clip-test-revision",
                            "stable_weight_files": [],
                        },
                        {
                            "role": "captioning",
                            "model_id": (
                                "florence-community/Florence-2-base"
                            ),
                            "repository_revision": "florence-test-revision",
                            "stable_weight_files": [],
                        },
                        {
                            "role": "ocr",
                            "model_id": (
                                "com.apple.Vision.VNRecognizeTextRequest"
                                if system == "Darwin"
                                else "PaddleOCR/default-English-pipeline"
                            ),
                            "repository_revision": None,
                            "stable_weight_files": [],
                        },
                    ]
                },
            },
            {
                "event": "exit",
                "stage": "proposal.rescue_shortlist",
                "payload": {
                    "candidates": [
                        {
                            "timestamp": 21.0,
                            "proposal_lane": "transition",
                            "transition_side": "post",
                        }
                    ]
                },
            },
            {
                "event": "exit",
                "stage": "selection.retained_after_alt",
                "payload": {
                    "candidates": [
                        {
                            "timestamp": 33.0,
                            "structured_delta_categories": [
                                "blank_populated",
                                "date",
                                "status",
                            ],
                        }
                    ]
                },
            },
        ],
    }
    release_evidence._atomic_write_json(frames / "pipeline_trace.json", trace)
    release_evidence._atomic_write_json(
        frames / "debug_qa_trace.json",
        {"schema": 1, "targets": []},
    )
    (
        artifacts,
        targets,
        budgets,
        redundancy,
        failures,
    ) = release_evidence._collect_artifact_evidence(bundle, contract)
    assert failures == []

    if source_kind == "git":
        source_identity = {
            "kind": "git",
            "commit_sha": source_value,
            "version": version,
        }
        locations = {
            "interpreter": "/test/bin/python",
            "environment_root": "/test",
            "base_prefix": "/base",
            "working_directory": "/test/source",
            "keyframe_module_path": "/test/source/keyframe/__init__.py",
            "keyframe_package_root": "/test/source/keyframe",
            "distribution_root": None,
            "isolated": False,
            "repository_pythonpath_present": False,
            "repository_shadowing": False,
            "package_inside_environment": None,
            "distribution_inside_environment": None,
        }
    elif source_kind == "artifact":
        artifact = bundle / "source" / f"keyframe-{version}-py3-none-any.whl"
        artifact.parent.mkdir()
        artifact.write_bytes(source_value.encode("utf-8"))
        source_identity = {
            "kind": "artifact",
            "name": "keyframe",
            "version": version,
            "filename": artifact.name,
            "path": artifact.relative_to(bundle).as_posix(),
            "size_bytes": artifact.stat().st_size,
            "sha256": release_evidence._sha256(artifact),
        }
        locations = {
            "interpreter": "/test/bin/python",
            "environment_root": "/test",
            "base_prefix": "/base",
            "working_directory": "/tmp/keyframe-release-runtime",
            "keyframe_module_path": "/test/lib/python/site-packages/keyframe/__init__.py",
            "keyframe_package_root": "/test/lib/python/site-packages/keyframe",
            "distribution_root": "/test/lib/python/site-packages",
            "isolated": True,
            "repository_pythonpath_present": False,
            "repository_shadowing": False,
            "package_inside_environment": True,
            "distribution_inside_environment": True,
            "installation_archive_filename": artifact.name,
            "installation_archive_sha256": source_identity["sha256"],
        }
    else:
        raise ValueError(source_kind)

    packages = {
        "keyframe": version,
        "pyobjc_framework_vision": "11.0" if system == "Darwin" else None,
        "paddleocr": "3.7.0" if system == "Linux" else None,
        "paddlepaddle": "3.3.1" if system == "Linux" else None,
    }
    probe = {
        "system": system,
        "machine": machine,
        "platform_release": "test",
        "python": "3.12.13",
        "packages": packages,
    }
    report = {
        "identifier": release_evidence.EVIDENCE_IDENTIFIER,
        "schema_version": release_evidence.EVIDENCE_SCHEMA_VERSION,
        "source_identity": source_identity,
        "fixture": release_evidence._fixture_record(contract, bundle),
        "artifacts": artifacts,
        "configuration": release_evidence._configuration_record(qa_path, bundle),
        "targets": targets,
        "budgets": budgets,
        "redundancy": redundancy,
        "platform": release_evidence._platform_record(probe),
        "packages": packages,
        "ocr_backend": release_evidence._ocr_record(probe),
        "model_provenance": release_evidence._model_provenance(trace),
        "package_locations": locations,
        "qualification": release_evidence._trace_qualification(trace),
        "validation": {"passed": True, "failures": []},
    }
    release_evidence._atomic_write_json(bundle / "evidence.json", report)
    assert release_evidence.replay_bundle(bundle).passed
    return report


def write_cross_platform_frame_evidence(
    root: Path,
    *,
    source_kind: str = "git",
    source_value: str = "a" * 40,
    version: str = "0.6.3",
) -> dict:
    darwin_bundle = root / "frame-evidence" / "darwin-arm64"
    linux_bundle = root / "frame-evidence" / "linux-x86_64"
    darwin = write_frame_evidence_bundle(
        darwin_bundle,
        system="Darwin",
        machine="arm64",
        source_kind=source_kind,
        source_value=source_value,
        version=version,
    )
    linux = write_frame_evidence_bundle(
        linux_bundle,
        system="Linux",
        machine="x86_64",
        source_kind=source_kind,
        source_value=source_value,
        version=version,
    )
    entries = {}
    for name, bundle, report in (
        ("darwin_arm64", darwin_bundle, darwin),
        ("linux_x86_64", linux_bundle, linux),
    ):
        report_path = bundle / "evidence.json"
        entries[name] = {
            "report_path": report_path.relative_to(root).as_posix(),
            "report_sha256": release_evidence._sha256(report_path),
            "evidence": report,
        }
    return {
        "identifier": "keyframe.cross-platform-frame-evidence",
        "schema_version": 1,
        **entries,
    }
