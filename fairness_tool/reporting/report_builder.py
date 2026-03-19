from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os

def get_tailored_recommendations(selections, metrics_before, metrics_after):
    recs = []
    
    # 1. Fairness Gap Analysis
    di_before = metrics_before.get("Disparate Impact", 1.0)
    di_after = metrics_after.get("Disparate Impact", 1.0)
    spd_after = metrics_after.get("Statistical Parity Difference", 0.0)
    
    # Check if DI is still far from 1 (using 0.8-1.25 rule)
    if (di_after < 0.8 or di_after > 1.25) or abs(spd_after) > 0.1:
        if abs(di_after - 1.0) < abs(di_before - 1.0):
            recs.append("<b>Fairness Gap Remaining:</b> Mitigation improved parity, but a significant gap still exists. Consider combining pre-processing with an <i>in-processing</i> algorithm (like Adversarial Debiasing) to target the model's internal weights.")
        else:
            recs.append("<b>Low Mitigation Impact:</b> The current method had minimal effect on fairness. This often happens if the bias is deeply embedded in features rather than just label frequency. Try <i>Relabeling</i> if you used Resampling, or audit for 'proxy' features.")

    # 2. Performance Trade-off Analysis
    acc_before = metrics_before.get("Test Accuracy", 0.0)
    acc_after = metrics_after.get("Test Accuracy", 0.0)
    if acc_before > 0.5 and (acc_before - acc_after) > 0.05:
        recs.append(f"<b>Significant Accuracy Trade-off:</b> Accuracy dropped from {acc_before:.2f} to {acc_after:.2f}. The current mitigation might be too aggressive. Try reducing resampling ratios or using a more complex model (like Gradient Boosting) that might better handle the fairer data distribution.")

    # 3. Preprocessing Audit
    prep = selections.get('preprocessing', {})
    if "Drop" in prep.get('missing_values', ''):
         recs.append("<b>Missing Data Bias:</b> You dropped rows with missing values. If one sensitive group had more missing data (common in marginalized communities), you might have introduced 'exclusion bias' before the analysis even started.")
    
    if "Label encoding" in prep.get('encoding', ''):
        recs.append("<b>Encoding Choice:</b> Label encoding was used. This can impose an arbitrary order on categorical groups that models like Logistic Regression interpret as magnitude. Switch to <i>One-Hot Encoding</i> for fairer feature representation.")

    # 4. Metric Specifics
    fair = selections.get('fairness', {})
    if "Demographic Parity" in fair.get('specific_metric', '') and abs(spd_after) < 0.05:
        recs.append("<b>Beyond Parity:</b> You achieved Demographic Parity. Next, evaluate <i>Equalized Odds</i> to ensure that among qualified candidates, the error rates are also balanced.")

    # Default if nothing specific triggered
    if not recs:
        recs.append("<b>Continuous Monitoring:</b> Your model maintains a good balance of accuracy and fairness. Implement automated 'Fairness Drift' detection to ensure this holds as the real-world data distribution changes.")
        
    return recs

def generate_pdf_report(filename, stats_before, metrics_before, metrics_after, plots=None, output_dir="outputs/reports", stats_after=None, selections=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    filepath = os.path.join(output_dir, filename)
    doc = SimpleDocTemplate(filepath, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    # Custom styles
    styles.add(ParagraphStyle(name='ItalicNormal', parent=styles['Normal'], fontName='Helvetica-Oblique'))
    styles.add(ParagraphStyle(name='Small', parent=styles['Normal'], fontSize=8, leading=10))
    
    story = []
    
    # 1. Title & Introduction
    story.append(Paragraph("Fairness Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("This report summarizes the fairness audit and bias mitigation process applied to your dataset. It includes a comparison of model performance and fairness metrics before and after mitigation.", styles['Normal']))
    story.append(Spacer(1, 18))
    
    # 2. Summary of Selections (The "What and Why")
    if selections:
        story.append(Paragraph("1. Configuration Summary", styles['Heading2']))
        story.append(Spacer(1, 6))
        
        # Preprocessing Explanations
        prep = selections.get('preprocessing', {})
        story.append(Paragraph("<b>Preprocessing Steps:</b>", styles['Normal']))
        story.append(Paragraph(f"• <b>Feature Selection:</b> {prep.get('feature_selection', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"• <b>Missing Values:</b> {prep.get('missing_values', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"• <b>Outlier Handling:</b> {prep.get('outliers', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"• <b>Primary Encoding:</b> {prep.get('encoding', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"• <b>Cleanup Encoding:</b> {prep.get('remaining_categoricals', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Fairness Setup
        fair = selections.get('fairness', {})
        mit = selections.get('mitigation', {})
        story.append(Paragraph("<b>Fairness Setup:</b>", styles['Normal']))
        story.append(Paragraph(f"• <b>Sensitive Attribute:</b> {fair.get('sensitive_column', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"• <b>Privileged Group:</b> {fair.get('privileged_group', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"• <b>Metric Chosen:</b> {fair.get('specific_metric', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"• <b>Bias Mitigation:</b> {mit.get('method', 'N/A')}", styles['Normal']))
        
        # Methodology Logic (Pros/Cons)
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Methodology Explained:</b>", styles['Normal']))
        
        metric_name = fair.get('specific_metric', '')
        if "Demographic Parity" in metric_name:
            story.append(Paragraph("<i>Demographic Parity</i> ensures the model predicts the positive outcome (e.g., 'Hired') at the same rate for all groups. <b>Pros:</b> Promotes equal representation. <b>Cons:</b> Can ignore differences in underlying qualifications (base rates).", styles['ItalicNormal']))
        elif "Equalized Odds" in metric_name:
            story.append(Paragraph("<i>Equalized Odds</i> ensures that the model is equally accurate for both groups (balancing False Positives and False Negatives). <b>Pros:</b> Punishes models that are 'lazier' or less accurate for minority groups. <b>Cons:</b> Harder to achieve simultaneously with accuracy.", styles['ItalicNormal']))
            
        mit_name = selections.get('mitigation', {}).get('method', '')
        if "Resampling" in mit_name:
            story.append(Paragraph("<i>Resampling mitigation</i> balances the dataset by duplicating minority group samples or removing majority ones. <b>Pros:</b> Simple and effective. <b>Cons:</b> Can lead to overfitting if too many samples are duplicated.", styles['ItalicNormal']))
        elif "Relabeling" in mit_name:
            story.append(Paragraph("<i>Relabeling mitigation</i> flips labels of individuals near the decision boundary to achieve parity. <b>Pros:</b> Directly targets the metric. <b>Cons:</b> Modifies the 'ground truth' of your data.", styles['ItalicNormal']))
        elif "Synthetic" in mit_name:
            if "CDA" in mit_name:
                 story.append(Paragraph("<i>Counterfactual Data Augmentation (CDA)</i> creates new samples by flipping the sensitive attribute (e.g., changing 'Male' to 'Female') while keeping other features constant. <b>Pros:</b> Forces the model to be invariant to the sensitive attribute. <b>Cons:</b> Can create unrealistic data points if features are highly correlated with the sensitive attribute.", styles['ItalicNormal']))
            elif "SMOTE" in mit_name:
                 story.append(Paragraph("<i>SMOTE (Synthetic Minority Over-sampling Technique)</i> generates synthetic samples for the minority class (or group) by interpolating between existing samples. <b>Pros:</b> Increases data diversity and balances classes. <b>Cons:</b> Can propagate noise if not applied carefully.", styles['ItalicNormal']))
            else:
                 story.append(Paragraph("<i>Synthetic Data Generation</i> creates new samples to improve balance. <b>Pros:</b> Increases data diversity. <b>Cons:</b> Relies on the quality of the generation method.", styles['ItalicNormal']))

        story.append(Spacer(1, 18))

    # 3. Dataset Statistics Comparison Table
    story.append(Paragraph("2. Dataset Statistics Comparison", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    stat_comp_data = [['Statistic', 'Baseline', 'Mitigated']]
    all_stat_keys_set = set(stats_before.keys()) | set((stats_after or {}).keys())
    
    preferred_order = ['Target Variable', 'Features Selected', 'Categorical Features', 'Numerical Features', 'Total Features', 'Total Samples', 'Missing Values']
    ordered_keys = [k for k in preferred_order if k in all_stat_keys_set]
    remaining_keys = sorted([k for k in all_stat_keys_set if k not in preferred_order])
    all_stat_keys = ordered_keys + remaining_keys
    
    for k in all_stat_keys:
        val_b = stats_before.get(k, 'N/A')
        val_m = (stats_after or {}).get(k, 'N/A')
        stat_comp_data.append([k, str(val_b), str(val_m)])
        
    t_stats = Table(stat_comp_data, colWidths=[200, 100, 100])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
    ]))
    story.append(t_stats)
    story.append(Spacer(1, 18))
    
    # 4. Comparison Table (Fairness & Performance)
    story.append(Paragraph("3. Fairness & Performance Results", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    comp_data = [['Metric', 'Baseline', 'Mitigated']]
    all_keys = set(metrics_before.keys()) | set(metrics_after.keys())
    
    for k in sorted(list(all_keys)):
        val_b = metrics_before.get(k, 'N/A')
        val_m = metrics_after.get(k, 'N/A')
        fmt_b = f"{val_b:.4f}" if isinstance(val_b, float) else str(val_b)
        fmt_m = f"{val_m:.4f}" if isinstance(val_m, float) else str(val_m)
        comp_data.append([k, fmt_b, fmt_m])
        
    t_comp = Table(comp_data, colWidths=[200, 100, 100])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Note:</b> A 'Statistical Parity Difference' closer to 0 and a 'Disparate Impact' closer to 1 indicate a fairer model.", styles['Small']))
    story.append(PageBreak())

    # 5. Visualizations with Explanations
    if plots:
        story.append(Paragraph("4. Visualization Analysis", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Legend/Explanations mapping
        plot_explanations = {
            "Baseline Confusion Matrix": "Shows how often the model correctly predicted the target in the original data. High diagonal values are good.",
            "Mitigated Confusion Matrix": "Shows model performance after mitigation. Compare this to the baseline to see if accuracy was sacrificed for fairness.",
            "Baseline Subgroup Confusion Matrices": "Confusion matrices broken down by sensitive group. Helps identify if errors are concentrated in specific groups.",
            "Mitigated Subgroup Confusion Matrices": "Subgroup confusion matrices after mitigation.",
            "Baseline KDE Predicted Probabilities": "Distribution of predicted probabilities per group. Large overlaps suggest high parity; separated peaks suggest bias.",
            "Mitigated KDE Predicted Probabilities": "Probability distributions after mitigation.",
            "Performance & Fairness Metrics Comparison": "A side-by-side view of all key metrics. Look for Fairness bars moving toward ideal values (0 for diffs, 1 for ratios).",
            "Class Distribution Comparison": "Shows if the balance between positive and negative labels changed after mitigation.",
            "Sensitive Attribute Distribution Comparison": "Shows if the representation of different sensitive groups (e.g. Races) was altered.",
            "Selection Rate Comparison P(Y=1|A)": "<b>Crucial:</b> Shows the probability of getting a positive outcome per group. Ideally, the bars should be level.",
            "Grouped Bar Charts (Positive Rate)": "Detailed view of subgroup outcome rates.",
            "Baseline Subgroup Heatmap": "Counts of individuals per (Group, Outcome) before mitigation.",
            "Mitigated Subgroup Heatmap": "Counts of individuals per (Group, Outcome) after mitigation."
        }
        
        for title, img_path in plots.items():
            if title == "Fairness-Utility Trade-off":
                continue
                
            if os.path.exists(img_path):
                story.append(Paragraph(title, styles['Heading3']))
                explanation = plot_explanations.get(title, "Visualization of dataset characteristics.")
                story.append(Paragraph(explanation, styles['Normal']))
                story.append(Spacer(1, 6))
                
                im = Image(img_path, width=400, height=300) 
                story.append(im)
                story.append(Spacer(1, 12))
            else:
                story.append(Paragraph(f"Image not found: {img_path}", styles['Normal']))

    # 6. Recommendations
    story.append(Spacer(1, 18))
    story.append(Paragraph("5. Recommendations for Next Iterations", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    # Sub-section: Strategic Suggestions (Dynamic)
    story.append(Paragraph("Strategic Suggestions (Data-Driven):", styles['Heading3']))
    recommendations = get_tailored_recommendations(selections, metrics_before, metrics_after)
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 4))
    
    story.append(Spacer(1, 12))
    
    # Sub-section: General Best Practices (Static)
    story.append(Paragraph("General Best Practices:", styles['Heading3']))
    general_tips = [
        "<b>Data Collection:</b> If selection rates are highly unequal, consider collecting more samples for the unprivileged group rather than just oversampling existing data to improve model generalization.",
        "<b>Feature Engineering:</b> Audit your features for 'proxies' (e.g., zip codes or interest-based data can often act as a proxy for race or gender) and consider removing them.",
        "<b>Ensemble Mitigation:</b> If one technique (e.g., Resampling) isn't enough, try combining it with a different model type or another mitigation method (hybrid approach).",
        "<b>Human-in-the-loop:</b> Use these metrics as a guide, not a final rule. Always involve domain experts to interpret why a model might be biased and the real-world impact of predictions."
    ]
    for tip in general_tips:
        story.append(Paragraph(f"• {tip}", styles['Normal']))
        story.append(Spacer(1, 4))

    doc.build(story)
    print(f"Report saved to {filepath}")
