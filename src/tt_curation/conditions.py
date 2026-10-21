"""The four experimental conditions of the MulTaBench curation pipeline, and the five learners it runs."""
from enum import StrEnum
from typing import Dict, Optional

import pandas as pd


class Condition(StrEnum):
    STRUCTURED = "structured"        # numerical/categorical features only
    UNSTRUCTURED = "unstructured"    # text features only, frozen E5
    JOINT_FROZEN = "joint_frozen"    # all features, frozen E5
    JOINT_TAR = "joint_tar"          # all features, E5 LoRA-tuned on the target (target-aware)


# The text encoder of each condition, as MulTaBench names it (multabench.e5.constants.TextEncoder). As in the paper's
# code, the text-only condition reads frozen embeddings: "txt: text embeddings only, no FT".
CONDITION_ENCODER: Dict[Condition, Optional[str]] = {
    Condition.STRUCTURED: None,
    Condition.UNSTRUCTURED: "e5-small",
    Condition.JOINT_FROZEN: "e5-small",
    Condition.JOINT_TAR: "e5-small-tar",
}

# The five learners the course requires, by MulTaBench's SHORT_NAME.
MODELS = {
    "tabm": "TabM",
    "cat": "CatBoost",
    "light": "LightGBM",
    "tabpfnv2": "TabPFN-v2",
    "tabpfnv2p5": "TabPFN-v2p5",
}


def condition_features(x: pd.DataFrame, text_columns: list, condition: Condition) -> pd.DataFrame:
    if condition == Condition.STRUCTURED:
        return x.drop(columns=text_columns)
    if condition == Condition.UNSTRUCTURED:
        return x[text_columns]
    return x
