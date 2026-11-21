"""Seed the checked-in synthetic corpus into a tenant, idempotently.

The default mode is a read-only dry run: it validates the corpus manifest,
reads the tenant's existing versions, and prints the safe action for every
document. Nothing is written unless ``--apply`` is passed explicitly.

Safety properties:

- Only documents declared with ``manage: "create"`` or
  ``manage: "repair_current"`` can be written. ``manage: "observe"`` never
  writes unless ``--materialize-observe`` is passed explicitly for a disposable
  test database, and then only when the observed document is entirely absent.
- A document whose intended content already exists as an older version is
  refused. Re-running the seeder can never make an older version current.
- No document, version, chunk, or embedding is ever deleted. Stored version
  history is never rewritten.
- The seeder refuses to run at all if any planned action is blocked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.corpus_plan import (
    CorpusManifest,
    CorpusPlanError,
    ExistingVersion,
    PlanAction,
    load_corpus_manifest,
    plan_corpus,
)

DEFAULT_MANIFEST = REPO_ROOT / "evals" / "corpus" / "synthetic_v1" / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the synthetic corpus manifest.",
    )
    parser.add_argument(
        "--tenant-id",
        required=True,
        help="Tenant UUID that receives the corpus.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the planned writes. Without this flag the seeder is read-only.",
    )
    parser.add_argument(
        "--materialize-observe",
        action="store_true",
        help=(
            "Also create absent 'observe' documents. Intended only for disposable "
            "test/CI databases that must reproduce the full evaluation corpus."
        ),
    )
    return parser


def read_inventory(tenant_id: UUID) -> dict[str, tuple[ExistingVersion, ...]]:
    """Read the tenant's existing versions without modifying anything."""
    from sqlalchemy import text

    from app.db.connections import get_engine

    with get_engine().connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                SELECT
                    d.source_key,
                    dv.version_number,
                    dv.content_hash,
                    dv.is_current,
                    d.status,
                    (
                        SELECT count(*) FROM chunks c
                        WHERE c.document_version_id = dv.id
                    ) AS chunk_count
                FROM documents d
                JOIN document_versions dv ON dv.document_id = d.id
                WHERE d.tenant_id = :tenant_id
                ORDER BY d.source_key, dv.version_number
                """
                ),
                {"tenant_id": tenant_id},
            )
            .mappings()
            .all()
        )

    inventory: dict[str, list[ExistingVersion]] = {}
    for row in rows:
        inventory.setdefault(row["source_key"], []).append(
            ExistingVersion(
                version_number=row["version_number"],
                content_hash=row["content_hash"],
                is_current=row["is_current"],
                chunk_count=row["chunk_count"],
                status=row["status"],
            )
        )
    return {source_key: tuple(versions) for source_key, versions in inventory.items()}


def tenant_exists(tenant_id: UUID) -> bool:
    from sqlalchemy import text

    from app.db.connections import get_engine

    with get_engine().connect() as connection:
        return (
            connection.execute(
                text("SELECT 1 FROM tenants WHERE id = :tenant_id"),
                {"tenant_id": tenant_id},
            ).scalar_one_or_none()
            is not None
        )


def _format_action(action: PlanAction) -> str:
    current = "absent" if action.current_version is None else f"v{action.current_version}"
    ingest = (
        "-"
        if not action.versions_to_ingest
        else ",".join(f"v{number}" for number in action.versions_to_ingest)
    )
    return (
        f"{action.source_key:<38} {action.kind:<28} "
        f"intended=v{action.intended_version} current={current} "
        f"ingest={ingest}  {action.reason}"
    )


def apply_actions(
    manifest: CorpusManifest,
    actions: tuple[PlanAction, ...],
    tenant_id: UUID,
) -> None:
    """Execute non-blocked actions through the existing ingestion service."""
    from app.core.auth import Principal
    from app.services.ingestion import ingest_document

    principal = Principal(
        tenant_id=tenant_id,
        subject="seed-synthetic-corpus",
        roles=frozenset({"document_writer"}),
    )

    for action in actions:
        if not action.is_write:
            continue

        document = manifest.by_source_key[action.source_key]
        versions = action.versions_to_ingest or (action.intended_version,)

        for version_number in versions:
            version = document.version(version_number)
            content = version.path.read_text(encoding="utf-8")
            result = ingest_document(
                principal=principal,
                tenant_id=tenant_id,
                source_key=document.source_key,
                title=document.title,
                source_type=document.source_type,
                content=content,
                authority_level=document.authority_level,
            )

            if result["content_hash"] != version.content_sha256:
                raise RuntimeError(
                    f"{document.source_key}: ingestion returned a different "
                    "content hash than the manifest declared"
                )
            if result["version_number"] != version_number:
                raise RuntimeError(
                    f"{document.source_key}: ingestion created version "
                    f"{result['version_number']} but the manifest expected "
                    f"version {version_number}"
                )


def verify_settled(
    manifest: CorpusManifest,
    tenant_id: UUID,
) -> dict[str, tuple[ExistingVersion, ...]]:
    """Re-read the inventory and prove every document settled on its intent."""
    inventory = read_inventory(tenant_id)
    remaining = plan_corpus(manifest, inventory)
    # ``observe`` documents are intentionally never written, so on a fresh
    # database they are legitimately absent. Everything else must have settled.
    settled_kinds = {"skip_current", "observe_missing"}
    unexpected = [action for action in remaining if action.kind not in settled_kinds]
    if unexpected:
        raise RuntimeError(
            "Corpus did not settle after apply; re-planning still reports: "
            + "; ".join(f"{action.source_key}={action.kind}" for action in unexpected)
        )
    return inventory


def materialize_observed_actions(
    manifest: CorpusManifest,
    actions: tuple[PlanAction, ...],
) -> tuple[PlanAction, ...]:
    """Create absent ``observe`` documents for disposable test fixtures.

    Production seeding must never write observed documents. CI and fresh local
    test databases, however, need the complete evaluation corpus that the live
    retrieval tests assert against. Only a document that is entirely absent is
    materialized; existing observed content is never rewritten or replaced.
    """

    extras: list[PlanAction] = []
    for action in actions:
        if action.kind != "observe_missing":
            continue
        document = manifest.by_source_key[action.source_key]
        extras.append(
            PlanAction(
                kind="create_versions",
                source_key=action.source_key,
                intended_version=document.intended_current_version,
                current_version=None,
                versions_to_ingest=(document.intended_current_version,),
                reason="materializing an absent observe document for a test fixture",
            )
        )
    return tuple(extras)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        tenant_id = UUID(args.tenant_id)
    except ValueError:
        print("--tenant-id must be a valid UUID.", file=sys.stderr)
        return 2

    try:
        manifest = load_corpus_manifest(args.manifest)
    except CorpusPlanError as exc:
        print(f"Corpus manifest is invalid: {exc}", file=sys.stderr)
        return 2

    if not tenant_exists(tenant_id):
        print(f"Tenant '{tenant_id}' does not exist.", file=sys.stderr)
        return 2

    inventory = read_inventory(tenant_id)
    actions = plan_corpus(manifest, inventory)
    if args.materialize_observe:
        actions = actions + materialize_observed_actions(manifest, actions)

    print(f"Corpus: {manifest.corpus_id} v{manifest.corpus_version}")
    print(f"Tenant: {tenant_id}")
    print(f"Mode:   {'APPLY' if args.apply else 'DRY RUN (no writes)'}")
    print()
    for action in actions:
        print(_format_action(action))

    blocked = [action for action in actions if action.is_blocked]
    if blocked:
        print()
        print(
            "Refusing to continue: the corpus has blocked actions. Resolve them "
            "manually before seeding.",
            file=sys.stderr,
        )
        return 2

    writes = [action for action in actions if action.is_write]
    if not args.apply:
        print()
        print(
            f"Dry run complete: {len(writes)} document(s) would be written, "
            f"{len(actions) - len(writes)} already settled. No database writes "
            "were performed."
        )
        return 0

    try:
        apply_actions(manifest, actions, tenant_id)
        verify_settled(manifest, tenant_id)
    except Exception as exc:  # noqa: BLE001 - report without leaking content
        print(f"Seeding failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print()
    print(
        f"Applied {len(writes)} document action(s) and verified the corpus "
        "settled on the intended current versions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
