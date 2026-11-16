"""Repo, storage, and package paths.

Layout::

    <git repo>/
      .git/
      storage/                     # required; payload bytes; not git-tracked
        raw_data/
        artifacts/                 # model/data payloads; not EDA output
            /script_stem
                /plots
                    /section_name
                /tables
      ccf_ml/src/ccf_ml/
        package_artifacts/         # collection and process configurations; git-tracked
          cv_splits/
      scripts/
      experiments/

The git repository is the project. Find ``.git`` by walking up from this
module. ``storage`` must be a directory inside that repo root. That layout
is not configurable.

Never use ``Path.cwd()``. Collection and process configuration folders are
static methods on those classes (``configs_dir``, ``process_configurations_dir``).
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_GIT_DIRNAME = ".git"
_STORAGE_DIRNAME = "storage"
_RAW_DATA_DIRNAME = "raw_data"
_ARTIFACTS_DIRNAME = "artifacts"
_PLOTS_DIRNAME = "plots"
_TABLES_DIRNAME = "tables"
_TRAIN_CSV = "train.csv"
_TEST_CSV = "test.csv"
_PACKAGE_ARTIFACTS_DIRNAME = "package_artifacts"
_CV_SPLITS_DIRNAME = "cv_splits"


def find_repo_root() -> Path:
    """Walk up from this module until ``.git`` appears."""
    current_dir = Path(__file__).resolve().parent
    while True:
        git_entry = current_dir / _GIT_DIRNAME
        if git_entry.exists():
            return current_dir
        parent_dir = current_dir.parent
        if parent_dir == current_dir:
            raise FileNotFoundError(
                f"Could not find '{_GIT_DIRNAME}' in any parent of {Path(__file__).resolve()}"
            )
        current_dir = parent_dir


def get_repo_root() -> Path:
    """Return the git repository root."""
    return find_repo_root()


def get_storage_dir() -> Path:
    """Return ``<repo>/storage``. Raises if it is missing or not inside the repo."""
    repo = get_repo_root().resolve()
    storage = repo / _STORAGE_DIRNAME
    if not storage.is_dir():
        raise FileNotFoundError(
            f"'{_STORAGE_DIRNAME}' must be a directory inside the git repo ({repo}): {storage}"
        )
    storage_resolved = storage.resolve()
    try:
        storage_resolved.relative_to(repo)
    except ValueError as exc:
        raise FileNotFoundError(
            f"'{_STORAGE_DIRNAME}' resolved outside the git repo ({repo}): {storage_resolved}"
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


def get_train_csv_path() -> Path:
    """Return ``<storage>/raw_data/train.csv``."""
    return get_raw_data_dir() / _TRAIN_CSV


def get_test_csv_path() -> Path:
    """Return ``<storage>/raw_data/test.csv``."""
    return get_raw_data_dir() / _TEST_CSV


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


def get_script_plot_dir(script_name: str | Path, *, section: str) -> Path:
    """Return ``<storage>/artifacts/<script_stem>/plots/<section>`` (created on demand)."""
    return _ensure_dir(
        get_artifacts_dir()
        / _script_stem(script_name)
        / _PLOTS_DIRNAME
        / _section_stem(section)
    )


def get_script_table_dir(script_name: str | Path) -> Path:
    """Return ``<storage>/artifacts/<script_stem>/tables`` (created on demand)."""
    return _ensure_dir(
        get_artifacts_dir() / _script_stem(script_name) / _TABLES_DIRNAME
    )


def get_package_dir() -> Path:
    """Return the ``ccf_ml`` package directory (this module's parent)."""
    return Path(__file__).resolve().parent


def get_package_artifacts_dir() -> Path:
    """Return ``<package>/package_artifacts``. Raises if it is missing."""
    path = get_package_dir() / _PACKAGE_ARTIFACTS_DIRNAME
    if not path.is_dir():
        raise FileNotFoundError(
            f"'{_PACKAGE_ARTIFACTS_DIRNAME}' must be a directory inside the "
            f"package ({get_package_dir()}): {path}"
        )
    return path


def get_cv_splits_dir() -> Path:
    """Return ``<package_artifacts>/cv_splits`` (created on demand)."""
    return _ensure_dir(get_package_artifacts_dir() / _CV_SPLITS_DIRNAME)
