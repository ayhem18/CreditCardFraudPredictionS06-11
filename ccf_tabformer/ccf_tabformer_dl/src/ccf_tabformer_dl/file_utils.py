"""Project, storage, and package paths.

Layout::

    <toolbox project>/
      dl_toolbox_project.json      # project marker (this tree, not the parent git repo)
      storage/                     # required; payload bytes; not git-tracked
        raw_data/                  # original Kaggle CSVs; generator handwriting
          train.csv
          test.csv
        data/
          corrected/               # working CSVs; extra singleton decimals dropped
            train.csv
            test.csv
            manifest.json
        artifacts/                 # experiment tables/plots; not training run payloads
            /script_stem
                /plots
                    /section_name
                /tables
            /<group>
                /script_stem
                    /plots
                    /tables
        data_for_training/         # frozen inner_table / row_subset numpy payloads
      ccf_tabformer_dl/src/ccf_tabformer_dl/
        package_artifacts/         # collection + process JSON (identity, git-tracked)
        data_contract/             # field_spec, vocabulary, inner_table, row_subset
        tabformer_components/      # TableSpec, embedder, masker, encoder, head, model
        task_engines/              # overfit / full-run engines
      scripts/
      experiments/

This project is nested inside the competition git repo. Find the toolbox
project by walking up from this module until ``dl_toolbox_project.json``
appears. ``storage`` must be a directory inside that project root. That
layout is not configurable.

Never use ``Path.cwd()``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.join import resolve_under_project
from dl_toolbox.base.hashing.hash_manager import HashManager

from ccf_tabformer_dl.project import load_project_layout

log = logging.getLogger(__name__)

_MARKER_FILENAME = "dl_toolbox_project.json"
_STORAGE_DIRNAME = "storage"
_RAW_DATA_DIRNAME = "raw_data"
_DATA_DIRNAME = "data"
_CORRECTED_DIRNAME = "corrected"
_ARTIFACTS_DIRNAME = "artifacts"
_PLOTS_DIRNAME = "plots"
_TABLES_DIRNAME = "tables"
_TRAIN_CSV = "train.csv"
_TEST_CSV = "test.csv"
_MANIFEST_FILENAME = "manifest.json"


def find_project_root() -> Path:
    """Walk up from this module until ``dl_toolbox_project.json`` appears."""
    current_dir = Path(__file__).resolve().parent
    while True:
        marker = current_dir / _MARKER_FILENAME
        if marker.is_file():
            return current_dir
        parent_dir = current_dir.parent
        if parent_dir == current_dir:
            raise FileNotFoundError(
                f"Could not find '{_MARKER_FILENAME}' in any parent of "
                f"{Path(__file__).resolve()}"
            )
        current_dir = parent_dir


def get_project_root() -> Path:
    """Return the toolbox project root (the directory that holds the marker)."""
    return find_project_root()


def get_storage_dir() -> Path:
    """Return ``<project>/storage``. Raises if it is missing or not inside the project."""
    project = get_project_root().resolve()
    storage = project / _STORAGE_DIRNAME
    if not storage.is_dir():
        raise FileNotFoundError(
            f"'{_STORAGE_DIRNAME}' must be a directory inside the project "
            f"({project}): {storage}"
        )
    storage_resolved = storage.resolve()
    try:
        storage_resolved.relative_to(project)
    except ValueError as exc:
        raise FileNotFoundError(
            f"'{_STORAGE_DIRNAME}' resolved outside the project ({project}): "
            f"{storage_resolved}"
        ) from exc
    return storage_resolved


def _ensure_dir(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.exists():
        raise FileNotFoundError(f"Expected a directory but found a file: {path}")
    log.warning("Creating missing directory: %s", path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_raw_data_dir() -> Path:
    """Return ``<storage>/raw_data`` (created on demand)."""
    return _ensure_dir(get_storage_dir() / _RAW_DATA_DIRNAME)


def get_raw_train_csv_path() -> Path:
    """Return ``<storage>/raw_data/train.csv`` (original Kaggle copy)."""
    return get_raw_data_dir() / _TRAIN_CSV


def get_raw_test_csv_path() -> Path:
    """Return ``<storage>/raw_data/test.csv`` (original Kaggle copy)."""
    return get_raw_data_dir() / _TEST_CSV


def get_corrected_data_dir() -> Path:
    """Return ``<storage>/data/corrected`` (created on demand)."""
    return _ensure_dir(get_storage_dir() / _DATA_DIRNAME / _CORRECTED_DIRNAME)


def get_corrected_train_csv_path() -> Path:
    """Return ``<storage>/data/corrected/train.csv`` (need not exist yet)."""
    return get_corrected_data_dir() / _TRAIN_CSV


def get_corrected_test_csv_path() -> Path:
    """Return ``<storage>/data/corrected/test.csv`` (need not exist yet)."""
    return get_corrected_data_dir() / _TEST_CSV


def get_corrected_manifest_path() -> Path:
    """Return ``<storage>/data/corrected/manifest.json``."""
    return get_corrected_data_dir() / _MANIFEST_FILENAME


def _require_csv(path: Path, *, label: str) -> Path:
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"{label} not found: {path}. "
        "Run experiments/data/discard_extra_decimals.py"
    )


def get_train_csv_path() -> Path:
    """Return ``<storage>/data/corrected/train.csv`` (working data)."""
    return _require_csv(get_corrected_train_csv_path(), label="corrected train CSV")


def get_test_csv_path() -> Path:
    """Return ``<storage>/data/corrected/test.csv`` (working data)."""
    return _require_csv(get_corrected_test_csv_path(), label="corrected test CSV")


def _script_stem(script_name: str | Path) -> str:
    stem = Path(script_name).stem.strip()
    if not stem or stem in {".", ".."}:
        raise ValueError(f"script name is empty or invalid: {script_name!r}")
    return stem


def get_artifacts_dir() -> Path:
    """Return ``<storage>/artifacts`` (created on demand)."""
    return _ensure_dir(get_storage_dir() / _ARTIFACTS_DIRNAME)


def _section_stem(section: str) -> str:
    stem = section.strip()
    if not stem or stem in {".", ".."}:
        raise ValueError(f"section name is empty or invalid: {section!r}")
    if "/" in stem or "\\" in stem:
        raise ValueError(f"section name must be a single path component: {section!r}")
    return stem


def get_script_plot_dir(
    script_name: str | Path, *, section: str, group: str | None = None
) -> Path:
    """Return the plot dir for one script (created on demand).

    Without ``group``: ``<storage>/artifacts/<script_stem>/plots/<section>``.
    With ``group`` (the experiments folder the script lives in):
    ``<storage>/artifacts/<group>/<script_stem>/plots/<section>`` — one
    folder per notebook inside one folder per experiment group, instead of
    a flat artifacts root.
    """
    base = get_artifacts_dir()
    if group is not None:
        base = base / _section_stem(group)
    return _ensure_dir(
        base
        / _script_stem(script_name)
        / _PLOTS_DIRNAME
        / _section_stem(section)
    )


def get_script_table_dir(
    script_name: str | Path, *, group: str | None = None
) -> Path:
    """Return the table dir for one script (created on demand).

    ``group`` nests the script under its experiments folder, mirroring
    :func:`get_script_plot_dir`.
    """
    base = get_artifacts_dir()
    if group is not None:
        base = base / _section_stem(group)
    return _ensure_dir(base / _script_stem(script_name) / _TABLES_DIRNAME)


def get_package_dir() -> Path:
    """Return the ``ccf_tabformer_dl`` package directory (this module's parent)."""
    return Path(__file__).resolve().parent


def package_artifacts_dir() -> Path:
    """Return ``<package>/package_artifacts`` (created on demand)."""
    return resolve_under_project(
        get_package_dir(),
        "package_artifacts",
        label="package_artifacts",
    )


def package_artifact_directory(relative_path: str) -> Directory:
    """Package-artifact folder. Nested path may be missing."""
    artifacts = Directory.from_absolute(
        path=package_artifacts_dir(),
        policy=DirectoryParsePolicy.CREATE,
    )
    return Directory.from_under(
        root=artifacts,
        relative_path=relative_path,
        policy=DirectoryParsePolicy.ALLOW_MISSING,
    )


def project_hash_manager() -> HashManager:
    """Project ``HashManager`` from ``dl_toolbox_project.json`` / ``identity.json``."""
    return load_project_layout(
        start=Path(__file__), tracking_backend=None
    ).hash_manager


def data_for_training_dir() -> Path:
    """``{project_root}/storage/data_for_training``. Need not exist yet."""
    return resolve_under_project(
        get_project_root(),
        "storage/data_for_training",
        label="data_for_training",
    )
