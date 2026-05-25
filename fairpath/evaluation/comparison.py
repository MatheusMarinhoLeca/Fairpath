import pandas as pd

def compare_metrics(baseline_metrics, mitigated_metrics):
    comparison = pd.DataFrame([baseline_metrics, mitigated_metrics], index=['Baseline', 'Mitigated']).T
    comparison['Difference'] = comparison['Mitigated'] - comparison['Baseline']
    return comparison
