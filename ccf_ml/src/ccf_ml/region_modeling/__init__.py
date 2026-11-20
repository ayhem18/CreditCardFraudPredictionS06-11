"""No-alarm region modeling: profile-stratified CV and variant ladder.

Separated from the main pipeline until this direction proves itself; merge
later if it does. Also hosts the unified alarm x profile x target CV
(``alarm_profile_cv``) - one fold assignment over all rows shared by the
global and region trainers so their OOF numbers are comparable.
"""

from ccf_ml.region_modeling.profiles import (
    add_device_group,
    add_merchant_tier,
    add_profile_id,
    build_no_alarm_frame,
    profile_stratum,
    with_profile_columns,
)
from ccf_ml.region_modeling.profile_cv import (
    ProfileStratifiedCv,
    build_profile_stratified_cv,
    dump_profile_stratified_cv,
    dump_to_path,
    fold_summaries,
    load_profile_stratified_cv,
    parse_profile_stratified_cv,
)
from ccf_ml.region_modeling.alarm_profile_cv import (
    AlarmProfileStratifiedCv,
    alarm_profile_cv_path,
    alarm_profile_stratum,
    alarm_target_totals,
    build_alarm_profile_stratified_cv,
    dump_alarm_profile_stratified_cv,
    fold_summaries as alarm_profile_fold_summaries,
    load_alarm_profile_stratified_cv,
    parse_alarm_profile_stratified_cv,
    restrict_to_no_alarm,
    write_alarm_profile_stratified_cv,
)
from ccf_ml.region_modeling.variants import (
    REGION_MODEL_VARIANTS,
    RegionVariantSpec,
    validate_variant,
    variant_by_name,
)
from ccf_ml.region_modeling.features import (
    add_profile_quantiles,
    add_profile_quantiles_vs_train,
    variant_model_matrix,
)

__all__ = [
    "AlarmProfileStratifiedCv",
    "ProfileStratifiedCv",
    "RegionVariantSpec",
    "REGION_MODEL_VARIANTS",
    "add_device_group",
    "add_merchant_tier",
    "add_profile_id",
    "add_profile_quantiles",
    "add_profile_quantiles_vs_train",
    "alarm_profile_cv_path",
    "alarm_profile_fold_summaries",
    "alarm_profile_stratum",
    "alarm_target_totals",
    "build_alarm_profile_stratified_cv",
    "build_no_alarm_frame",
    "build_profile_stratified_cv",
    "dump_alarm_profile_stratified_cv",
    "dump_profile_stratified_cv",
    "dump_to_path",
    "fold_summaries",
    "load_alarm_profile_stratified_cv",
    "load_profile_stratified_cv",
    "parse_alarm_profile_stratified_cv",
    "parse_profile_stratified_cv",
    "profile_stratum",
    "restrict_to_no_alarm",
    "validate_variant",
    "variant_by_name",
    "variant_model_matrix",
    "with_profile_columns",
    "write_alarm_profile_stratified_cv",
]
