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
            /dataset_versions
                /version_name
                    manifest.json
                    folds.json
                    data.parquet
            /training
                /exp_name
                    config.json
                    oof.csv
                    models/
                    submission.csv
                    eval_off/
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
_DATASET_VERSIONS_DIRNAME = "dataset_versions"
_DATASET_VERSION_MANIFEST_FILENAME = "manifest.json"
_DATASET_VERSION_FOLDS_FILENAME = "folds.json"
_DATASET_VERSION_DATA_FILENAME = "data.parquet"
_TRAINING_DIRNAME = "training"
_TRAINING_CONFIG_FILENAME = "config.json"
_TRAINING_OOF_FILENAME = "oof.csv"
_TRAINING_EVAL_DIRNAME = "eval_off"
_TRAINING_MODELS_DIRNAME = "models"
_TRAINING_SUBMISSION_FILENAME = "submission.csv"


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


def get_dataset_versions_dir() -> Path:
    """Return ``<storage>/artifacts/dataset_versions`` (created on demand)."""
    return _ensure_dir(get_artifacts_dir() / _DATASET_VERSIONS_DIRNAME)


def claim_dataset_version_dir(version_name: str) -> Path:
    """Create ``<dataset_versions>/<version_name>``. Raises if it exists."""
    name = _section_stem(version_name)
    path = get_dataset_versions_dir() / name
    if path.exists():
        raise FileExistsError(
            f"dataset version {name!r} already exists: {path}"
        )
    path.mkdir(parents=False, exist_ok=False)
    return path


def get_dataset_version_dir(version_name: str) -> Path:
    """Return an existing ``<dataset_versions>/<version_name>`` directory."""
    name = _section_stem(version_name)
    path = get_dataset_versions_dir() / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"dataset version {name!r} not found: {path}"
        )
    return path


def get_dataset_version_manifest_path(version_name: str) -> Path:
    """Return ``<dataset_versions>/<version_name>/manifest.json``."""
    return get_dataset_version_dir(version_name) / _DATASET_VERSION_MANIFEST_FILENAME


def get_dataset_version_folds_path(version_name: str) -> Path:
    """Return ``<dataset_versions>/<version_name>/folds.json``."""
    return get_dataset_version_dir(version_name) / _DATASET_VERSION_FOLDS_FILENAME


def get_dataset_version_data_path(version_name: str) -> Path:
    """Return ``<dataset_versions>/<version_name>/data.parquet``."""
    return get_dataset_version_dir(version_name) / _DATASET_VERSION_DATA_FILENAME


def get_training_artifacts_dir() -> Path:
    """Return ``<storage>/artifacts/training`` (created on demand)."""
    return _ensure_dir(get_artifacts_dir() / _TRAINING_DIRNAME)


def claim_training_exp_dir(exp_name: str) -> Path:
    """Create ``<storage>/artifacts/training/<exp_name>``. Raises if it exists."""
    name = _section_stem(exp_name)
    path = get_training_artifacts_dir() / name
    if path.exists():
        raise FileExistsError(
            f"training experiment {name!r} already exists: {path}"
        )
    path.mkdir(parents=False, exist_ok=False)
    return path


def get_training_exp_dir(exp_name: str) -> Path:
    """Return an existing ``<storage>/artifacts/training/<exp_name>`` directory."""
    name = _section_stem(exp_name)
    path = get_training_artifacts_dir() / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"training experiment {name!r} not found: {path}"
        )
    return path


def get_training_config_path(exp_name: str) -> Path:
    """Return ``<training>/<exp_name>/config.json``."""
    return get_training_exp_dir(exp_name) / _TRAINING_CONFIG_FILENAME


def get_training_oof_path(exp_name: str) -> Path:
    """Return ``<training>/<exp_name>/oof.csv``."""
    return get_training_exp_dir(exp_name) / _TRAINING_OOF_FILENAME


def get_training_eval_dir(exp_name: str) -> Path:
    """Return ``<training>/<exp_name>/eval_off`` (created on demand)."""
    return _ensure_dir(get_training_exp_dir(exp_name) / _TRAINING_EVAL_DIRNAME)


def get_training_models_dir(exp_name: str) -> Path:
    """Return ``<training>/<exp_name>/models`` (created on demand)."""
    return _ensure_dir(get_training_exp_dir(exp_name) / _TRAINING_MODELS_DIRNAME)


def get_training_fold_model_path(exp_name: str, *, fold_id: int) -> Path:
    """Return ``<training>/<exp_name>/models/fold_<fold_id>.joblib``."""
    if type(fold_id) is not int or fold_id < 0:
        raise ValueError(f"fold_id must be an int >= 0, got {fold_id!r}")
    return get_training_models_dir(exp_name) / f"fold_{fold_id}.joblib"


def get_training_submission_path(exp_name: str) -> Path:
    """Return ``<training>/<exp_name>/submission.csv``."""
    return get_training_exp_dir(exp_name) / _TRAINING_SUBMISSION_FILENAME
