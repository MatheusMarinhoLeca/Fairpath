# Configuration defaults

# List of common sensitive attributes to auto-detect
POTENTIAL_SENSITIVE_ATTRIBUTES = ['race', 'gender', 'sex', 'age', 'ethnicity', 'religion', 'disability', 'sexual_orientation', 'native_country']

# Default sensitive attribute to prioritize if found (per user requirement)
DEFAULT_PRIORITY_SENSITIVE = "Race"

# Internal column name for fairness evaluation preservation
FAIRNESS_EVAL_COL = "_fairness_eval_sens_attr"
