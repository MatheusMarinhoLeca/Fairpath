# Configuration defaults

# List of common sensitive attributes to auto-detect
POTENTIAL_SENSITIVE_ATTRIBUTES = ['race', 'gender', 'sex', 'age', 'ethnicity', 'religion', 'disability', 'sexual_orientation', 'native_country']

# Default sensitive attribute to prioritize if found (per user requirement)
DEFAULT_PRIORITY_SENSITIVE = "Race"