"""Dataset versions: named column-drop ablations on frozen flag-stratified folds.

A version is a name bound to a column-drop spec, the frozen flag-stratified
partition, and the resulting frame on disk::

    storage/artifacts/dataset_versions/<name>/
        manifest.json
        folds.json
        data.parquet

Build with :func:`build_dataset_version`, load with
:func:`load_dataset_version`, iterate folds with :func:`iter_folds`.
"""

from ccf_ml.dataset_versions.constants import TIME_COLUMNS
from ccf_ml.dataset_versions.builder import build_dataset_version
from ccf_ml.dataset_versions.loader import (
    DatasetVersion,
    VersionFold,
    dataset_version_exists,
    iter_folds,
    list_dataset_versions,
    load_dataset_version,
    load_fold,
)
from ccf_ml.dataset_versions.manifest import (
    DatasetVersionManifest,
    load_dataset_version_manifest,
    sha256_file,
)
from ccf_ml.dataset_versions.specs import (
    DATASET_VERSION_SPECS,
    DatasetVersionSpec,
    validate_spec,
)
