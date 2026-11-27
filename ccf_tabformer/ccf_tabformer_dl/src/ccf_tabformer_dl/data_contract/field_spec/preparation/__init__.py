"""field_spec preparation oracle."""

from ccf_tabformer_dl.data_contract.field_spec.preparation.oracle import (
    build_field_spec_preparation,
)
from ccf_tabformer_dl.data_contract.field_spec.preparation.preparation import FieldSpecPreparation

__all__ = ["FieldSpecPreparation", "build_field_spec_preparation"]
