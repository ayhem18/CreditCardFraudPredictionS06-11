"""Cross-validation: alarm × target stratum, stratified k-fold, frozen splits.

The alarm-flag union is a high-rate region. A four-level stratum (alarm flag ×
``is_fraud``) is the same composition as running ``StratifiedKFold`` on the
flagged rows and on the unflagged rows separately, then concatenating fold
``i``. sklearn ``StratifiedKFold`` on the combined stratum does that in one
split.

The frozen artifact stores val ids per fold. Train ids are the complement.
"""

from ccf_ml.cross_validation.alarm import (
    alarm_flag_mask,
    binary_pair_stratum,
    flag_target_stratum,
)
from ccf_ml.cross_validation.constants import (
    ALARM_FLAG_COLUMNS,
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
    FLAG_STRATIFIED_CV_FILENAME,
    ID_COLUMN,
    TARGET_COLUMN,
)
from ccf_ml.cross_validation.flag_stratified import (
    FlagStratifiedCv,
    FoldSummary,
    build_flag_stratified_cv,
    flag_stratified_cv_path,
    load_flag_stratified_cv,
    write_flag_stratified_cv,
)
from ccf_ml.cross_validation.predefined import (
    predefined_split,
    predefined_split_from_folds,
    val_fold_vector,
)
from ccf_ml.cross_validation.stratified import Fold, stratified_kfold_ids
