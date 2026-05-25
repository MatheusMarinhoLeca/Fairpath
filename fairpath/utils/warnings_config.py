import warnings
import logging

def configure_warnings():
    """Centralized configuration for suppressing and managing environment warnings.
    
    This utility ensures that known noisy warnings from third-party libraries 
    (e.g., aif360, matplotlib) are handled consistently across the application 
    without cluttering the domain logic.
    """
    # 1. Standard library and third-party warnings
    warnings.simplefilter('default')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    
    # aif360 specific runtime warnings (often due to small sample sizes in subgroups)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="aif360.metrics.classification_metric")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="aif360.datasets.binary_label_dataset")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="aif360.metrics.binary_label_dataset_metric")
    
    # Generic ignore for scalar divide/invalid value encountered in ML operations
    # (Typical when a subgroup has zero samples or zero positive outcomes)
    warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
    
    # pandas performance warnings (common during extensive data transformations)
    try:
        from pandas.errors import PerformanceWarning
        warnings.filterwarnings("ignore", category=PerformanceWarning)
    except ImportError:
        pass

    # 2. Integrate warnings into the logging system
    logging.captureWarnings(True)

    # 3. Explicitly silence specific logging noise from presentation libraries
    logging.getLogger('matplotlib.category').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
