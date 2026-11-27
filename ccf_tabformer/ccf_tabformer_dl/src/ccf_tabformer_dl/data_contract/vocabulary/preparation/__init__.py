"""vocabulary preparation oracle."""

from ccf_tabformer_dl.data_contract.vocabulary.preparation.oracle import (
    build_vocabulary_preparation,
)
from ccf_tabformer_dl.data_contract.vocabulary.preparation.preparation import VocabularyPreparation

__all__ = ["VocabularyPreparation", "build_vocabulary_preparation"]
