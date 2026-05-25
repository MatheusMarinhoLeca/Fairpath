import numpy as np
from scipy.stats import norm
from typing import Tuple

class StatisticalValidator:
    """Provides statistically grounded validation for fairness disparities."""
    
    @staticmethod
    def compute_p_value_proportions(n1: int, p1: float, n2: int, p2: float) -> float:
        """Computes the p-value for the difference between two proportions (Z-test).
        
        Args:
            n1: Sample size of group 1.
            p1: Proportion of positive outcomes in group 1.
            n2: Sample size of group 2.
            p2: Proportion of positive outcomes in group 2.
            
        Returns:
            float: Two-tailed p-value.
        """
        # Pooled proportion
        count1 = n1 * p1
        count2 = n2 * p2
        p_pooled = (count1 + count2) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        if se == 0:
            return 1.0
            
        # Z-score
        z = (p1 - p2) / se
        
        # Two-tailed p-value
        p_value = 2 * (1 - norm.cdf(abs(z)))
        return p_value

    @staticmethod
    def is_disparity_significant(n1: int, p1: float, n2: int, p2: float, alpha: float = 0.05) -> Tuple[bool, float]:
        """Determines if the difference in proportions is statistically significant.
        
        Returns:
            (bool, float): (is_significant, p_value)
        """
        p_val = StatisticalValidator.compute_p_value_proportions(n1, p1, n2, p2)
        return p_val < alpha, p_val
