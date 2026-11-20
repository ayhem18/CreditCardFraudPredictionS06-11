"""The dataset version manifest: a strict JSON document, one per version."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

from ccf_ml.dataset_versions.constants import (
    DATASET_VERSION_MANIFEST_FILENAME,
    DATASET_VERSION_SCHEME,
)
from ccf_ml.dataset_versions.specs import validate_version_name
from ccf_ml.file_utils import get_dataset_version_manifest_path

_MANIFEST_KEYS = (
    "scheme",
    "name",
    "description",
    "drop_columns",
    "kept_columns",
    "id_column",
    "target_column",
    "n_rows",
    "n_columns",
    "n_splits",
    "cv_scheme",
    "cv_filename",
    "cv_sha256",
    "data_filename",
    "data_sha256",
)


class DatasetVersionManifest(NamedTuple):
    """What a dataset version is, and fingerprints of its two payload files.

    ``data_sha256`` covers ``data.parquet``; ``cv_sha256`` covers
    ``folds.json``. The loader verifies both by default.
    """

    scheme: str
    name: str
    description: str
    drop_columns: tuple[str, ...]
    kept_columns: tuple[str, ...]
    id_column: str
    target_column: str
    n_rows: int
    n_columns: int
    n_splits: int
    cv_scheme: str
    cv_filename: str
    cv_sha256: str
    data_filename: str
    data_sha256: str


def dump_dataset_version_manifest(
    document: DatasetVersionManifest,
) -> dict[str, object]:
    """JSON-ready manifest document."""
    return {
        "scheme": document.scheme,
        "name": document.name,
        "description": document.description,
        "drop_columns": list(document.drop_columns),
        "kept_columns": list(document.kept_columns),
        "id_column": document.id_column,
        "target_column": document.target_column,
        "n_rows": document.n_rows,
        "n_columns": document.n_columns,
        "n_splits": document.n_splits,
        "cv_scheme": document.cv_scheme,
        "cv_filename": document.cv_filename,
        "cv_sha256": document.cv_sha256,
        "data_filename": document.data_filename,
        "data_sha256": document.data_sha256,
    }


def parse_dataset_version_manifest(payload: object) -> DatasetVersionManifest:
    """Rebuild the manifest from a JSON object."""
    document = _require_dict(payload, name="dataset_version_manifest")
    _require_exact_keys(document, _MANIFEST_KEYS, name="dataset_version_manifest")
    scheme = _require_str(document["scheme"], name="scheme")
    if scheme != DATASET_VERSION_SCHEME:
        raise ValueError(
            f"scheme must be {DATASET_VERSION_SCHEME!r}, got {scheme!r}"
        )
    name = _require_str(document["name"], name="name")
    validate_version_name(name)
    description = document["description"]
    if not isinstance(description, str):
        raise TypeError(
            f"description must be a str, got {type(description)!r}"
        )
    return DatasetVersionManifest(
        scheme=scheme,
        name=name,
        description=description,
        drop_columns=_require_str_tuple(document["drop_columns"], name="drop_columns"),
        kept_columns=_require_str_tuple(document["kept_columns"], name="kept_columns"),
        id_column=_require_str(document["id_column"], name="id_column"),
        target_column=_require_str(document["target_column"], name="target_column"),
        n_rows=_require_int(document["n_rows"], name="n_rows"),
        n_columns=_require_int(document["n_columns"], name="n_columns"),
        n_splits=_require_int(document["n_splits"], name="n_splits"),
        cv_scheme=_require_str(document["cv_scheme"], name="cv_scheme"),
        cv_filename=_require_str(document["cv_filename"], name="cv_filename"),
        cv_sha256=_require_sha256(document["cv_sha256"], name="cv_sha256"),
        data_filename=_require_str(document["data_filename"], name="data_filename"),
        data_sha256=_require_sha256(document["data_sha256"], name="data_sha256"),
    )


def write_dataset_version_manifest(
    document: DatasetVersionManifest,
    *,
    path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the manifest. Refuses to replace an existing file unless asked."""
    dest = path if path is not None else get_dataset_version_manifest_path(document.name)
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"dataset version manifest already exists at {dest}. "
            f"Pass overwrite=True to replace it."
        )
    text = json.dumps(
        dump_dataset_version_manifest(document),
        separators=(",", ":"),
        allow_nan=False,
    )
    dest.write_text(text, encoding="utf-8")
    return dest


def load_dataset_version_manifest(
    version_name: str,
    *,
    path: Path | None = None,
) -> DatasetVersionManifest:
    """Load the manifest for ``version_name`` from disk."""
    source = (
        path
        if path is not None
        else get_dataset_version_manifest_path(validate_version_name(version_name))
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    document = parse_dataset_version_manifest(payload)
    if document.name != validate_version_name(version_name):
        raise ValueError(
            f"manifest at {source} is for version {document.name!r}, "
            f"not {version_name!r}"
        )
    return document


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of ``path``'s bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_dict(payload: object, *, name: str) -> dict:
    if type(payload) is not dict:
        raise TypeError(f"{name} must be a dict, got {type(payload)!r}")
    return payload


def _require_exact_keys(
    payload: dict,
    keys: tuple[str, ...],
    *,
    name: str,
) -> None:
    expected = set(keys)
    actual = set(payload)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{name} keys do not match: missing={missing!r} extra={extra!r}"
        )


def _require_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value)!r}")
    return value


def _require_str(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty str, got {value!r}")
    return value


def _require_str_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a list, got {type(value)!r}")
    items: list[str] = []
    for i, item in enumerate(value):
        items.append(_require_str(item, name=f"{name}[{i}]"))
    return tuple(items)


def _require_sha256(value: object, *, name: str) -> str:
    text = _require_str(value, name=name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be 64 hex characters, got {text!r}")
    return text
