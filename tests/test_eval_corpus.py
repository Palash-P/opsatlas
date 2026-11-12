"""Validate the checked-in synthetic corpus and the idempotent seeding plan."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from evals.corpus_plan import (
    CREDENTIAL_PATTERNS,
    CorpusPlanError,
    ExistingVersion,
    load_corpus_manifest,
    plan_corpus,
    validate_corpus_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "evals" / "corpus" / "synthetic_v1"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"


@pytest.fixture(scope="module")
def manifest():
    return load_corpus_manifest(MANIFEST_PATH)


def _assert_no_credentials(text: str) -> None:
    for name, pattern in CREDENTIAL_PATTERNS:
        assert pattern.search(text) is None, f"credential-like pattern: {name}"


def test_manifest_loads_and_every_hash_matches(manifest) -> None:
    assert manifest.corpus_id == "synthetic_v1"
    assert manifest.corpus_version == 1
    assert len(manifest.documents) == 10

    for document in manifest.documents:
        assert document.synthetic is True
        assert document.intended_current_version == len(document.versions)
        for version in document.versions:
            assert version.path.is_file()
            assert len(version.content_sha256) == 64


def test_every_document_is_labelled_synthetic_in_content_or_manifest(manifest) -> None:
    readme = (CORPUS_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "every document in this directory is synthetic" in readme

    for document in manifest.documents:
        for version in document.versions:
            text = version.path.read_text(encoding="utf-8")
            if document.label_source == "content":
                assert "synthetic" in text[:400].lower()
            else:
                assert document.label_source == "manifest"


def test_corpus_contains_no_credentials_or_real_incident_claims(manifest) -> None:
    for document in manifest.documents:
        for version in document.versions:
            text = version.path.read_text(encoding="utf-8")
            _assert_no_credentials(text)
            assert "synthetic" in text.lower() or document.label_source == "manifest"


def test_corpus_has_the_expected_world_states(manifest) -> None:
    by_key = manifest.by_source_key

    assert set(by_key) == {
        "api-specs/payments-v3.md",
        "api-specs/payments.md",
        "postmortems/db-pool-2025-02.md",
        "runbooks/payment-api.md",
        "runbooks/payment-errors.md",
        "runbooks/postgres-pool.md",
        "runbooks/search-deploy.md",
        "runbooks/kafka-consumer-lag.md",
        "runbooks/redis-failover.md",
        "runbooks/incident-escalation.md",
    }

    multi_version = [
        document
        for document in manifest.documents
        if len(document.versions) > 1
    ]
    assert [document.source_key for document in multi_version] == [
        "api-specs/payments.md",
        "runbooks/payment-api.md",
    ]
    assert by_key["api-specs/payments.md"].intended_current_version == 2
    assert by_key["runbooks/payment-api.md"].intended_current_version == 2
    assert by_key["runbooks/payment-api.md"].manage == "repair_current"


def test_payments_migration_guide_has_a_genuine_version_transition(manifest) -> None:
    document = manifest.by_source_key["api-specs/payments.md"]
    v1 = document.version(1).path.read_text(encoding="utf-8")
    v2 = document.version(2).path.read_text(encoding="utf-8")

    assert "/api/v1/payments" in v1
    assert "deprecated" not in v1.lower()
    assert "/api/v3/payments" not in v1

    assert "/api/v1/payments" in v2
    assert "deprecated" in v2.lower()
    assert "/api/v3/payments" in v2


def test_payment_api_versions_differ_only_by_the_connectivity_step(manifest) -> None:
    document = manifest.by_source_key["runbooks/payment-api.md"]
    v1_lines = document.version(1).path.read_text(encoding="utf-8").splitlines()
    v2_lines = document.version(2).path.read_text(encoding="utf-8").splitlines()

    added = [
        line for line in v2_lines if line not in v1_lines
    ]
    removed = [line for line in v1_lines if line not in v2_lines]

    assert added == [
        "4. Verify database connectivity before retrying the payment request."
    ]
    assert removed == []


def test_validate_corpus_file_rejects_missing_synthetic_label(tmp_path) -> None:
    path = tmp_path / "runbook.md"
    path.write_text("# Runbook\nNothing to see here.\n", encoding="utf-8")

    with pytest.raises(CorpusPlanError, match="must state that they are synthetic"):
        validate_corpus_file(path, require_content_label=True)

    assert validate_corpus_file(path, require_content_label=False)


def test_validate_corpus_file_rejects_credentials(tmp_path) -> None:
    path = tmp_path / "runbook.md"
    path.write_text(
        "# Synthetic Runbook\nConnect with postgres://user:secret@db.example/app\n",
        encoding="utf-8",
    )

    with pytest.raises(CorpusPlanError, match="credential-like pattern"):
        validate_corpus_file(path, require_content_label=True)


def test_manifest_rejects_hash_mismatch(tmp_path) -> None:
    document = tmp_path / "doc.md"
    document.write_text("# Synthetic document\nBody.\n", encoding="utf-8")
    manifest = {
        "corpus_version": 1,
        "corpus_id": "broken",
        "documents": [
            {
                "source_key": "doc.md",
                "title": "Synthetic",
                "source_type": "runbook",
                "authority_level": 50,
                "synthetic": True,
                "label_source": "content",
                "manage": "create",
                "intended_current_version": 1,
                "versions": [
                    {
                        "version_number": 1,
                        "path": "doc.md",
                        "content_sha256": "0" * 64,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CorpusPlanError, match="declared content_sha256 does not match"):
        load_corpus_manifest(path)


def _settled_inventory(manifest) -> dict[str, tuple[ExistingVersion, ...]]:
    return {
        document.source_key: tuple(
            ExistingVersion(
                version_number=version.version_number,
                content_hash=version.content_sha256,
                is_current=version.version_number
                == document.intended_current_version,
                chunk_count=1,
            )
            for version in document.versions
        )
        for document in manifest.documents
    }


def test_new_document_creates_all_versions_in_order(manifest) -> None:
    actions = plan_corpus(manifest, {})
    payments = next(
        action
        for action in actions
        if action.source_key == "api-specs/payments.md"
    )

    assert payments.kind == "create_versions"
    assert payments.versions_to_ingest == (1, 2)
    assert payments.current_version is None


def test_repair_reactivates_the_existing_newer_version(manifest) -> None:
    document = manifest.by_source_key["runbooks/payment-api.md"]
    inventory = {
        document.source_key: (
            ExistingVersion(
                version_number=1,
                content_hash=document.version(1).content_sha256,
                is_current=True,
            ),
            ExistingVersion(
                version_number=2,
                content_hash=document.version(2).content_sha256,
                is_current=False,
            ),
        )
    }

    action = next(
        action
        for action in plan_corpus(manifest, inventory)
        if action.source_key == "runbooks/payment-api.md"
    )

    assert action.source_key == "runbooks/payment-api.md"
    assert action.kind == "reactivate_existing"
    assert action.versions_to_ingest == (2,)
    assert action.current_version == 1


def _apply_plan_to_inventory(manifest, actions, inventory):
    """Simulate ingestion's current-pointer behaviour for a plan."""
    updated = {
        source_key: list(versions) for source_key, versions in inventory.items()
    }
    for action in actions:
        if not action.is_write:
            continue
        document = manifest.by_source_key[action.source_key]
        versions = updated.setdefault(action.source_key, [])
        by_number = {version.version_number: version for version in versions}
        for number in action.versions_to_ingest:
            if number not in by_number:
                versions.append(
                    ExistingVersion(
                        version_number=number,
                        content_hash=document.version(number).content_sha256,
                        is_current=False,
                    )
                )
        target = max(action.versions_to_ingest)
        updated[action.source_key] = [
            ExistingVersion(
                version_number=version.version_number,
                content_hash=version.content_hash,
                is_current=version.version_number == target,
                chunk_count=1,
            )
            for version in sorted(versions, key=lambda item: item.version_number)
        ]
    return {
        source_key: tuple(versions)
        for source_key, versions in updated.items()
    }


def test_seeder_is_idempotent_and_never_repoints_to_an_older_version(
    manifest,
) -> None:
    payment_api = manifest.by_source_key["runbooks/payment-api.md"]
    initial = _settled_inventory(manifest)
    for document in manifest.documents:
        if document.manage == "create":
            initial.pop(document.source_key, None)
    initial[payment_api.source_key] = (
        ExistingVersion(
            version_number=1,
            content_hash=payment_api.version(1).content_sha256,
            is_current=True,
        ),
        ExistingVersion(
            version_number=2,
            content_hash=payment_api.version(2).content_sha256,
            is_current=False,
        ),
    )

    first_run = plan_corpus(manifest, initial)
    first_payment_action = next(
        action
        for action in first_run
        if action.source_key == payment_api.source_key
    )
    assert first_payment_action.versions_to_ingest == (2,)

    after_first_run = _apply_plan_to_inventory(manifest, first_run, initial)
    second_run = plan_corpus(manifest, after_first_run)

    assert {action.kind for action in second_run} == {"skip_current"}
    assert not any(action.is_write for action in second_run)

    for document in manifest.documents:
        versions = after_first_run[document.source_key]
        current = [
            version.version_number for version in versions if version.is_current
        ]
        assert current == [document.intended_current_version]


def test_settled_inventory_replans_without_writes(manifest) -> None:
    settled = _settled_inventory(manifest)

    replanned = plan_corpus(manifest, settled)

    assert {action.kind for action in replanned} == {"skip_current"}
    assert not any(action.is_write for action in replanned)


def test_seeder_refuses_to_make_an_older_version_current(manifest) -> None:
    document = manifest.by_source_key["api-specs/payments.md"]
    inventory = {
        document.source_key: (
            ExistingVersion(
                version_number=1,
                content_hash=document.version(1).content_sha256,
                is_current=False,
            ),
            ExistingVersion(
                version_number=2,
                content_hash=document.version(2).content_sha256,
                is_current=False,
            ),
            ExistingVersion(
                version_number=3,
                content_hash="f" * 64,
                is_current=True,
            ),
        )
    }

    action = next(
        action
        for action in plan_corpus(manifest, inventory)
        if action.source_key == "api-specs/payments.md"
    )

    assert action.kind == "refuse_downgrade"
    assert action.is_blocked is True
    assert action.versions_to_ingest == ()


def test_observe_mode_never_writes(manifest) -> None:
    document = manifest.by_source_key["runbooks/postgres-pool.md"]
    inventory = {
        document.source_key: (
            ExistingVersion(
                version_number=1,
                content_hash="e" * 64,
                is_current=True,
            ),
        )
    }

    action = next(
        action
        for action in plan_corpus(manifest, inventory)
        if action.source_key == "runbooks/postgres-pool.md"
    )

    assert action.kind == "observe_drift"
    assert action.is_write is False


def test_seeder_refuses_an_unexpected_version_number(manifest) -> None:
    document = manifest.by_source_key["api-specs/payments.md"]
    inventory = {
        document.source_key: (
            ExistingVersion(
                version_number=1,
                content_hash=document.version(1).content_sha256,
                is_current=True,
            ),
            ExistingVersion(
                version_number=2,
                content_hash="f" * 64,
                is_current=False,
            ),
        )
    }

    action = next(
        action
        for action in plan_corpus(manifest, inventory)
        if action.source_key == "api-specs/payments.md"
    )

    assert action.kind == "refuse_missing_earlier_versions" or (
        action.kind == "refuse_unexpected_version"
    )
    assert action.is_blocked is True


def test_seeder_cli_defaults_to_dry_run() -> None:
    from scripts.seed_synthetic_corpus import build_parser

    args = build_parser().parse_args(["--tenant-id", str(uuid4())])

    assert args.apply is False
    assert args.manifest.name == "manifest.json"


def test_seeder_cli_rejects_an_invalid_manifest_without_touching_the_database(
    tmp_path,
    capsys,
) -> None:
    from scripts.seed_synthetic_corpus import main

    exit_code = main(
        [
            "--tenant-id",
            str(uuid4()),
            "--manifest",
            str(tmp_path / "missing.json"),
        ]
    )

    assert exit_code == 2
    assert "Corpus manifest is invalid" in capsys.readouterr().err
