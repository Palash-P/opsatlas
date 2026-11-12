"""Pure corpus-manifest validation and idempotent ingestion planning.

This module performs no database access and no writes. It answers one question:
given a declared corpus manifest and the versions that already exist in a
tenant, what is the safe next action for each document?

The central invariant is that re-running a seeder must never make an older
version current. Planning therefore distinguishes "the intended content is
already current" (skip) from "the intended content exists as an older version"
(refuse) and from "the intended content is a genuinely newer version"
(create or reactivate).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.chunking import normalize_document_text

CORPUS_SCHEMA_VERSION = 1
MANAGE_MODES = frozenset({"observe", "create", "repair_current"})
LABEL_SOURCES = frozenset({"content", "manifest"})
SYNTHETIC_MARKER_WINDOW = 400
SYNTHETIC_README = "README.md"
SYNTHETIC_README_MARKER = "every document in this directory is synthetic"

# Patterns that must never appear in a checked-in synthetic corpus document.
# These are intentionally narrow to avoid flagging ordinary prose such as
# "the database password must never be logged".
CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("database_url", re.compile(r"\b(?:postgres|postgresql|mysql|redis)://\S+")),
    ("assigned_password", re.compile(r"(?i)\bpassword\s*[:=]\s*\S+")),
    ("assigned_api_key", re.compile(r"(?i)\bapi[_-]?key\s*[:=]\s*\S{8,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._-]{20,}")),
)


class CorpusPlanError(ValueError):
    """Raised when a corpus manifest or its files are invalid."""


def sha256_text(value: str) -> str:
    """Hash normalized document text exactly as ingestion does."""
    return hashlib.sha256(normalize_document_text(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CorpusVersion:
    version_number: int
    path: Path
    content_sha256: str


@dataclass(frozen=True)
class CorpusDocument:
    source_key: str
    title: str
    source_type: str
    authority_level: int
    synthetic: bool
    label_source: str
    manage: str
    intended_current_version: int
    versions: tuple[CorpusVersion, ...]

    def version(self, version_number: int) -> CorpusVersion:
        for version in self.versions:
            if version.version_number == version_number:
                return version
        raise CorpusPlanError(
            f"{self.source_key}: no declared version {version_number}"
        )


@dataclass(frozen=True)
class CorpusManifest:
    corpus_id: str
    corpus_version: int
    path: Path
    documents: tuple[CorpusDocument, ...]

    @property
    def by_source_key(self) -> dict[str, CorpusDocument]:
        return {document.source_key: document for document in self.documents}


@dataclass(frozen=True)
class ExistingVersion:
    version_number: int
    content_hash: str
    is_current: bool
    chunk_count: int = 0
    status: str = "active"


@dataclass(frozen=True)
class PlanAction:
    kind: str
    source_key: str
    intended_version: int
    current_version: int | None
    versions_to_ingest: tuple[int, ...] = ()
    reason: str = ""

    @property
    def is_write(self) -> bool:
        return self.kind in {"create_versions", "reactivate_existing"}

    @property
    def is_blocked(self) -> bool:
        return self.kind.startswith("refuse_")


def _fail(location: str, message: str) -> CorpusPlanError:
    return CorpusPlanError(f"{location}: {message}")


def _require_string(payload: dict[str, Any], field: str, location: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise _fail(location, f"'{field}' must be a nonempty string")
    return value


def _require_int(
    payload: dict[str, Any],
    field: str,
    location: str,
    *,
    minimum: int,
) -> int:
    value = payload.get(field)
    if type(value) is not int or value < minimum:
        raise _fail(location, f"'{field}' must be an integer >= {minimum}")
    return value


def _check_no_credentials(text: str, location: str, findings: list[str]) -> None:
    for name, pattern in CREDENTIAL_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            findings.append(
                f"{location}: contains a credential-like pattern ({name})"
            )


def validate_corpus_file(
    path: Path,
    *,
    require_content_label: bool = True,
) -> str:
    """Validate one corpus file and return its normalized content hash."""
    if not path.is_file():
        raise CorpusPlanError(f"Corpus file does not exist: '{path}'")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise CorpusPlanError(f"Corpus file is not valid UTF-8: '{path}'") from None
    except OSError as exc:
        raise CorpusPlanError(
            f"Cannot read corpus file '{path}': {exc.strerror or type(exc).__name__}"
        ) from None

    normalized = normalize_document_text(content)
    if not normalized:
        raise CorpusPlanError(f"Corpus file is empty: '{path}'")

    if require_content_label and (
        "synthetic" not in normalized[:SYNTHETIC_MARKER_WINDOW].lower()
    ):
        raise CorpusPlanError(
            f"'{path}': synthetic corpus documents must state that they are "
            f"synthetic within the first {SYNTHETIC_MARKER_WINDOW} characters, "
            "or the manifest must declare label_source 'manifest' with a "
            "synthetic notice in the corpus README"
        )

    findings: list[str] = []
    _check_no_credentials(normalized, str(path), findings)
    if findings:
        raise CorpusPlanError("; ".join(findings))

    return sha256_text(normalized)


def load_corpus_manifest(path: Path | str) -> CorpusManifest:
    """Load and fully validate a corpus manifest and every referenced file."""
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CorpusPlanError(
            f"Cannot read corpus manifest '{manifest_path}': "
            f"{exc.strerror or type(exc).__name__}"
        ) from None
    except json.JSONDecodeError as exc:
        raise _fail(str(manifest_path), f"invalid JSON: {exc.msg}") from None

    if not isinstance(payload, dict):
        raise _fail(str(manifest_path), "manifest must be a JSON object")

    corpus_version = _require_int(
        payload, "corpus_version", str(manifest_path), minimum=1
    )
    corpus_id = _require_string(payload, "corpus_id", str(manifest_path))
    documents_payload = payload.get("documents")
    if not isinstance(documents_payload, list) or not documents_payload:
        raise _fail(str(manifest_path), "'documents' must be a nonempty list")

    documents: list[CorpusDocument] = []
    seen_source_keys: set[str] = set()

    for index, entry in enumerate(documents_payload):
        location = f"{manifest_path}: documents[{index}]"
        if not isinstance(entry, dict):
            raise _fail(location, "document entry must be an object")

        source_key = _require_string(entry, "source_key", location)
        if source_key in seen_source_keys:
            raise _fail(location, f"duplicate source_key '{source_key}'")
        seen_source_keys.add(source_key)

        title = _require_string(entry, "title", location)
        source_type = _require_string(entry, "source_type", location)
        authority_level = _require_int(
            entry, "authority_level", location, minimum=0
        )
        if authority_level > 100:
            raise _fail(location, "'authority_level' must be <= 100")

        synthetic = entry.get("synthetic")
        if type(synthetic) is not bool:
            raise _fail(location, "'synthetic' must be a JSON boolean")
        if not synthetic:
            raise _fail(
                location,
                "every checked-in corpus document must be marked 'synthetic': true",
            )

        manage = _require_string(entry, "manage", location)
        if manage not in MANAGE_MODES:
            raise _fail(
                location,
                f"'manage' must be one of: {', '.join(sorted(MANAGE_MODES))}",
            )

        label_source = entry.get("label_source", "content")
        if label_source not in LABEL_SOURCES:
            raise _fail(
                location,
                f"'label_source' must be one of: {', '.join(sorted(LABEL_SOURCES))}",
            )

        intended_current_version = _require_int(
            entry, "intended_current_version", location, minimum=1
        )

        versions_payload = entry.get("versions")
        if not isinstance(versions_payload, list) or not versions_payload:
            raise _fail(location, "'versions' must be a nonempty list")

        versions: list[CorpusVersion] = []
        seen_numbers: set[int] = set()
        for version_entry in versions_payload:
            if not isinstance(version_entry, dict):
                raise _fail(location, "every version entry must be an object")

            version_number = _require_int(
                version_entry, "version_number", location, minimum=1
            )
            if version_number in seen_numbers:
                raise _fail(
                    location, f"duplicate version_number {version_number}"
                )
            seen_numbers.add(version_number)

            relative_path = _require_string(version_entry, "path", location)
            declared_hash = _require_string(
                version_entry, "content_sha256", location
            )
            if len(declared_hash) != 64 or any(
                character not in "0123456789abcdef" for character in declared_hash
            ):
                raise _fail(
                    location,
                    f"version {version_number}: 'content_sha256' must be 64 "
                    "lowercase hex characters",
                )

            version_path = manifest_path.parent / relative_path
            actual_hash = validate_corpus_file(
                version_path,
                require_content_label=label_source == "content",
            )
            if actual_hash != declared_hash:
                raise _fail(
                    location,
                    f"version {version_number}: declared content_sha256 does not "
                    f"match '{version_path}' (actual {actual_hash})",
                )

            versions.append(
                CorpusVersion(
                    version_number=version_number,
                    path=version_path,
                    content_sha256=declared_hash,
                )
            )

        versions.sort(key=lambda version: version.version_number)
        numbers = [version.version_number for version in versions]
        if numbers != list(range(1, len(numbers) + 1)):
            raise _fail(
                location,
                "declared versions must be numbered consecutively from 1 "
                f"(found {numbers})",
            )

        if intended_current_version != numbers[-1]:
            raise _fail(
                location,
                "'intended_current_version' must equal the highest declared "
                f"version ({numbers[-1]})",
            )

        documents.append(
            CorpusDocument(
                source_key=source_key,
                title=title,
                source_type=source_type,
                authority_level=authority_level,
                synthetic=synthetic,
                label_source=label_source,
                manage=manage,
                intended_current_version=intended_current_version,
                versions=tuple(versions),
            )
        )

    if any(document.label_source == "manifest" for document in documents):
        readme_path = manifest_path.parent / SYNTHETIC_README
        if not readme_path.is_file():
            raise _fail(
                str(manifest_path),
                f"documents declare label_source 'manifest' but "
                f"'{readme_path}' is missing",
            )
        readme_text = readme_path.read_text(encoding="utf-8").lower()
        if SYNTHETIC_README_MARKER not in readme_text:
            raise _fail(
                str(readme_path),
                f"the corpus README must contain the notice "
                f"'{SYNTHETIC_README_MARKER}'",
            )

    return CorpusManifest(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        path=manifest_path,
        documents=tuple(documents),
    )


def _current(existing: tuple[ExistingVersion, ...]) -> ExistingVersion | None:
    for version in existing:
        if version.is_current:
            return version
    return None


def plan_document(
    document: CorpusDocument,
    existing: tuple[ExistingVersion, ...],
) -> PlanAction:
    """Decide the safe next action for one document."""
    intended = document.version(document.intended_current_version)
    current = _current(existing)
    current_number = current.version_number if current is not None else None

    if current is not None and current.content_hash == intended.content_sha256:
        return PlanAction(
            kind="skip_current",
            source_key=document.source_key,
            intended_version=document.intended_current_version,
            current_version=current_number,
            reason="intended content is already the current version",
        )

    if document.manage == "observe":
        if current is None:
            return PlanAction(
                kind="observe_missing",
                source_key=document.source_key,
                intended_version=document.intended_current_version,
                current_version=None,
                reason="document is not present; observe mode never writes",
            )
        return PlanAction(
            kind="observe_drift",
            source_key=document.source_key,
            intended_version=document.intended_current_version,
            current_version=current_number,
            reason=(
                "current content hash differs from the manifest; observe mode "
                "never writes"
            ),
        )

    if not existing:
        return PlanAction(
            kind="create_versions",
            source_key=document.source_key,
            intended_version=document.intended_current_version,
            current_version=None,
            versions_to_ingest=tuple(
                version.version_number for version in document.versions
            ),
            reason="document is absent; create declared versions in order",
        )

    by_hash = {version.content_hash: version for version in existing}
    existing_target = by_hash.get(intended.content_sha256)
    if existing_target is not None:
        if current is not None and (
            existing_target.version_number < current.version_number
        ):
            return PlanAction(
                kind="refuse_downgrade",
                source_key=document.source_key,
                intended_version=document.intended_current_version,
                current_version=current_number,
                reason=(
                    f"intended content is version {existing_target.version_number} "
                    f"but version {current.version_number} is current; refusing to "
                    "make an older version current"
                ),
            )

        return PlanAction(
            kind="reactivate_existing",
            source_key=document.source_key,
            intended_version=existing_target.version_number,
            current_version=current_number,
            versions_to_ingest=(existing_target.version_number,),
            reason=(
                "intended content already exists as a newer immutable version; "
                "reactivate it without rewriting history"
            ),
        )

    highest_existing = max(version.version_number for version in existing)
    if document.intended_current_version != highest_existing + 1:
        return PlanAction(
            kind="refuse_unexpected_version",
            source_key=document.source_key,
            intended_version=document.intended_current_version,
            current_version=current_number,
            reason=(
                f"intended version {document.intended_current_version} would not "
                f"be the next version after {highest_existing}; refusing to create "
                "an unexpected version number"
            ),
        )

    existing_numbers = {version.version_number for version in existing}
    missing_earlier = [
        version.version_number
        for version in document.versions
        if version.version_number < document.intended_current_version
        and version.version_number not in existing_numbers
    ]
    if missing_earlier:
        return PlanAction(
            kind="refuse_missing_earlier_versions",
            source_key=document.source_key,
            intended_version=document.intended_current_version,
            current_version=current_number,
            reason=(
                "cannot create version "
                f"{document.intended_current_version} before earlier versions "
                f"{missing_earlier} exist"
            ),
        )

    return PlanAction(
        kind="create_versions",
        source_key=document.source_key,
        intended_version=document.intended_current_version,
        current_version=current_number,
        versions_to_ingest=(document.intended_current_version,),
        reason="intended content is a new next version",
    )


def plan_corpus(
    manifest: CorpusManifest,
    inventory: dict[str, tuple[ExistingVersion, ...]],
) -> tuple[PlanAction, ...]:
    """Plan every document, preserving manifest order for deterministic output."""
    return tuple(
        plan_document(
            document,
            inventory.get(document.source_key, ()),
        )
        for document in manifest.documents
    )
