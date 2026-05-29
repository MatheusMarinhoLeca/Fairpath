from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass(frozen=True)
class DatasetSchema:
    """Metadata registry for standard datasets to ensure semantic integrity."""
    name: str
    target_col: str
    positive_label: Any
    sensitive_cols: List[str]
    privileged_values: Dict[str, Any]
    description: str = ""

# Standard Dataset Registry
DATASET_REGISTRY = {
    'Adult Census Income': DatasetSchema(
        name='Adult Census Income',
        target_col='income',
        positive_label='>50K',
        sensitive_cols=['race', 'sex'],
        privileged_values={'race': 'White', 'sex': 'Male'},
        description="Predict whether income exceeds $50K/yr based on census data."
    ),
    'ProPublica COMPAS': DatasetSchema(
        name='ProPublica COMPAS',
        target_col='two_year_recid',
        positive_label='Survived',  # 'Survived' means "did not recidivate" (favorable outcome)
        sensitive_cols=['sex', 'race'],
        privileged_values={'sex': 'Female', 'race': 'Caucasian'},
        description="Predict 2-year recidivism risk. Outcome 'Survived' (No Recidivism) is favorable."
    ),
    'German Credit': DatasetSchema(
        name='German Credit',
        target_col='credit',
        positive_label='good', # 'good' means "good credit"
        sensitive_cols=['sex', 'age_cat'],
        privileged_values={'sex': 'male', 'age_cat': 'aged'},
        description="Classify people as good or bad credit risks."
    )
}
