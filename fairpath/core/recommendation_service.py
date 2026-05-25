from typing import List, Dict, Any
from fairpath.core.models import Recommendation
from fairpath.utils.statistics import StatisticalValidator

class RecommendationService:
    """Generates adaptive, evidence-based fairness recommendations grounded in statistical analysis."""

    def generate_recommendations(self, 
                                 selections: Dict[str, Any], 
                                 stats_before: Dict[str, Any],
                                 metrics_before: Dict[str, Any], 
                                 metrics_after: Dict[str, Any]) -> List[Recommendation]:
        """Orchestrates the generation of data-driven advice by comparing baseline and mitigated results."""
        recs = []
        
        # 1. Fairness & Statistical Audit (Adaptive)
        recs.extend(self._audit_fairness_evolution(stats_before, metrics_before, metrics_after))
        
        # 2. Performance Trade-off Audit
        recs.extend(self._audit_performance_tradeoffs(metrics_before, metrics_after))
        
        # 3. Methodological Audit
        recs.extend(self._audit_methodology(selections))
        
        return recs

    def _audit_fairness_evolution(self, stats: Dict[str, Any], metrics_b: Dict[str, Any], metrics_a: Dict[str, Any]) -> List[Recommendation]:
        recs = []
        
        # Helper to find the "worst" value across all groups for a specific metric
        def get_max_gap(metrics: Dict[str, Any], base_name: str, ideal: float = 0.0) -> float:
            vals = [v for k, v in metrics.items() if base_name in k]
            if not vals:
                return ideal
            # Find the value furthest from ideal
            return max(vals, key=lambda x: abs(x - ideal))

        # 1. Parity Audit (Demographic Parity)
        di_b = get_max_gap(metrics_b, "Disparate Impact", ideal=1.0)
        di_a = get_max_gap(metrics_a, "Disparate Impact", ideal=1.0)
        spd_b = get_max_gap(metrics_b, "Statistical Parity Difference", ideal=0.0)
        spd_a = get_max_gap(metrics_a, "Statistical Parity Difference", ideal=0.0)

        # 2. Error Rate Audit (Equalized Odds)
        eod_b = get_max_gap(metrics_b, "Equal Opportunity Difference", ideal=0.0)
        eod_a = get_max_gap(metrics_a, "Equal Opportunity Difference", ideal=0.0)
        aod_b = get_max_gap(metrics_b, "Average Odds Difference", ideal=0.0)
        aod_a = get_max_gap(metrics_a, "Average Odds Difference", ideal=0.0)

        # Check for remaining gap in Demographic Parity
        if (di_a < 0.8 or di_a > 1.25) or abs(spd_a) > 0.1:
            if abs(di_a - 1.0) < abs(di_b - 1.0) or abs(spd_a) < abs(spd_b):
                recs.append(Recommendation(
                    category="Fairness Gap Remaining",
                    description="Mitigation improved parity, but a significant gap still exists.",
                    evidence=f"Largest Statistical Parity Difference is {spd_a:.4f}.",
                    action="Mitigation improved parity, but a significant gap still exists. Consider combining pre-processing with an in-processing algorithm (like Adversarial Debiasing) to target the model's internal weights.",
                    confidence_level="High"
                ))
            else:
                recs.append(Recommendation(
                    category="Low Mitigation Impact",
                    description="The current method had minimal effect on fairness.",
                    evidence=f"Statistical Parity Difference remains high at {spd_a:.4f}.",
                    action="The current method had minimal effect on fairness. This often happens if the bias is deeply embedded in features rather than just label frequency. Try a different mitigation method or audit for 'proxy' features.",
                    confidence_level="High"
                ))
        elif (abs(spd_b) > 0.1 or di_b < 0.8 or di_b > 1.25):
             recs.append(Recommendation(
                category="Beyond Parity",
                description="Successfully achieved Demographic Parity.",
                evidence=f"SPD dropped to {spd_a:.4f} and DI is {di_a:.4f}.",
                action="You achieved Demographic Parity! Next, evaluate Equalized Odds to ensure that among qualified candidates, the error rates (False Positives/Negatives) are also balanced.",
                confidence_level="High"
            ))
        
        # Check for remaining gap in Equalized Odds
        if abs(eod_a) > 0.1 or abs(aod_a) > 0.1:
            if abs(eod_a) < abs(eod_b) or abs(aod_a) < abs(aod_b):
                recs.append(Recommendation(
                    category="Error Rate Imbalance",
                    description="Model accuracy is still significantly different between groups.",
                    evidence=f"Equal Opportunity Difference is {eod_a:.4f}.",
                    action="The model is still failing 'qualified' individuals in the unprivileged group more often. Try 'Post-processing' mitigation (Calibrated Equalized Odds) which adjusts the decision threshold per group.",
                    confidence_level="High"
                ))
            else:
                recs.append(Recommendation(
                    category="Stubborn Error Bias",
                    description="Mitigation failed to balance error rates.",
                    evidence=f"Average Odds Difference is {aod_a:.4f}.",
                    action="Resampling rarely fixes Equalized Odds effectively. Consider using a 'Constraint-based' in-processing method or collecting more balanced high-quality data for the minority groups.",
                    confidence_level="High"
                ))
        elif (abs(eod_b) > 0.1 or abs(aod_b) > 0.1):
            recs.append(Recommendation(
                category="Equalized Odds Achieved",
                description="Successfully balanced error rates across groups.",
                evidence=f"EOD dropped to {eod_a:.4f}.",
                action="You have achieved Equalized Odds! This model is now equally accurate for all groups. Monitor performance over time to ensure this parity holds with new data.",
                confidence_level="High"
            ))

        return recs

    def _audit_performance_tradeoffs(self, metrics_b: Dict[str, Any], metrics_a: Dict[str, Any]) -> List[Recommendation]:
        recs = []
        acc_b = metrics_b.get("Test Accuracy", 0)
        acc_a = metrics_a.get("Test Accuracy", 0)
        
        if acc_b > 0 and (acc_b - acc_a) > 0.05:
            recs.append(Recommendation(
                category="Significant Accuracy Trade-off",
                description="Accuracy dropped significantly after mitigation.",
                evidence=f"Accuracy dropped by {acc_b - acc_a:.4f} ({acc_b:.4f} -> {acc_a:.4f})",
                action=f"Accuracy dropped from {acc_b:.2f} to {acc_a:.2f}. The current mitigation might be too aggressive. Try reducing resampling ratios or using a more complex model (like Gradient Boosting) that might better handle the fairer data distribution.",
                confidence_level="High"
            ))
        return recs

    def _audit_methodology(self, selections: Dict[str, Any]) -> List[Recommendation]:
        recs = []
        prep = selections.get('preprocessing', {})
        
        if "Drop" in prep.get('missing_values', ''):
            recs.append(Recommendation(
                category="Missing Data Bias",
                description="Exclusion bias risk due to 'Drop' strategy.",
                evidence="Rows with missing data were removed.",
                action="You dropped rows with missing values. If one sensitive group had more missing data, you might have introduced 'exclusion bias' before the analysis even started.",
                confidence_level="Medium"
            ))
            
        if "Label Encoding" in prep.get('encoding', ''):
             recs.append(Recommendation(
                category="Encoding Choice",
                description="Potential ordinal bias from Label Encoding.",
                evidence="Categorical features were mapped to integers.",
                action="Label encoding was used. This can impose an arbitrary order on categorical groups that models interpret as magnitude. Switch to One-Hot Encoding for fairer feature representation.",
                confidence_level="Medium"
            ))
            
        return recs
