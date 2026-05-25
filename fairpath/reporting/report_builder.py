from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os

def format_val(val):
    """Helper to round floats to 4 decimal places or return string."""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)

def generate_pdf_report(filename, stats_before, metrics_before, metrics_after, recommendations=None, plots=None, output_dir="outputs/reports", stats_after=None, selections=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    filepath = os.path.join(output_dir, filename)
    doc = SimpleDocTemplate(filepath, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    # Custom styles
    styles.add(ParagraphStyle(name='ItalicNormal', parent=styles['Normal'], fontName='Helvetica-Oblique'))
    styles.add(ParagraphStyle(name='Small', parent=styles['Normal'], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name='TableCell', parent=styles['Normal'], fontSize=9, leading=11))
    styles.add(ParagraphStyle(name='TableHeader', parent=styles['Normal'], fontSize=10, leading=12, fontName='Helvetica-Bold', textColor=colors.whitesmoke))
    
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
        
        # Preprocessing Steps
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
        story.append(Paragraph(f"• <b>Fairness Definition Chosen:</b> {fair.get('specific_metric', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"• <b>Bias Mitigation:</b> {mit.get('method', 'N/A')}", styles['Normal']))
        
        # Methodology Explained
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Methodology Explained:</b>", styles['Normal']))
        
        metric_name = fair.get('specific_metric', '')
        if "Demographic Parity" in metric_name:
            story.append(Paragraph("<i>Demographic Parity</i> ensures the model predicts the positive outcome (e.g., 'Hired') at the same rate for all groups. <b>Pros:</b> Promotes equal representation. <b>Cons:</b> Can ignore differences in underlying qualifications (base rates).", styles['ItalicNormal']))
        elif "Equalized Odds" in metric_name:
            story.append(Paragraph("<i>Equalized Odds</i> ensures that the model is equally accurate for both groups (balancing False Positives and False Negatives). <b>Pros:</b> Punishes models that are 'lazier' or less accurate for minority groups. <b>Cons:</b> Harder to achieve simultaneously with accuracy.", styles['ItalicNormal']))
            
        mit_name = mit.get('method', '')
        if "Resampling" in mit_name:
            story.append(Paragraph("<i>Resampling mitigation</i> balances the dataset by duplicating minority group samples or removing majority ones. <b>Pros:</b> Simple and effective. <b>Cons:</b> Can lead to overfitting if too many samples are duplicated.", styles['ItalicNormal']))
        elif "Relabeling" in mit_name:
            story.append(Paragraph("<i>Relabeling mitigation</i> flips labels of individuals near the decision boundary to achieve parity. <b>Pros:</b> Directly targets the metric. <b>Cons:</b> Modifies the 'ground truth' of your data.", styles['ItalicNormal']))
        elif "Synthetic" in mit_name:
             story.append(Paragraph("<i>Synthetic Data Generation</i> creates new samples to improve balance. <b>Pros:</b> Increases data diversity. <b>Cons:</b> Relies on the quality of the generation method.", styles['ItalicNormal']))

        story.append(Spacer(1, 18))

    # 3. Dataset Statistics Comparison Table
    story.append(Paragraph("2. Dataset Statistics Comparison", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    stat_comp_data = [[Paragraph("Statistic", styles['TableHeader']), Paragraph("Baseline", styles['TableHeader']), Paragraph("Mitigated", styles['TableHeader'])]]
    all_stat_keys_set = set(stats_before.keys()) | set((stats_after or {}).keys())
    
    preferred_order = ['Target Variable', 'Features Selected', 'Categorical Features', 'Numerical Features', 'Total Features', 'Total Samples', 'Missing Values']
    ordered_keys = [k for k in preferred_order if k in all_stat_keys_set]
    remaining_keys = sorted([k for k in all_stat_keys_set if k not in preferred_order])
    all_stat_keys = ordered_keys + remaining_keys
    
    for k in all_stat_keys:
        val_b = stats_before.get(k, 'N/A')
        val_m = (stats_after or {}).get(k, 'N/A')
        stat_comp_data.append([
            Paragraph(str(k), styles['TableCell']), 
            Paragraph(format_val(val_b), styles['TableCell']), 
            Paragraph(format_val(val_m), styles['TableCell'])
        ])
        
    t_stats = Table(stat_comp_data, colWidths=[240, 110, 110])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
    ]))
    story.append(t_stats)
    story.append(Spacer(1, 18))
    
    # 4. Comparison Table (Fairness & Performance)
    story.append(Paragraph("3. Fairness & Performance Results", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    comp_data = [[Paragraph("Metric", styles['TableHeader']), Paragraph("Baseline", styles['TableHeader']), Paragraph("Mitigated", styles['TableHeader'])]]
    all_keys = set(metrics_before.keys()) | set(metrics_after.keys())
    
    for k in sorted(list(all_keys)):
        val_b = metrics_before.get(k, 'N/A')
        val_m = metrics_after.get(k, 'N/A')
        comp_data.append([
            Paragraph(str(k), styles['TableCell']), 
            Paragraph(format_val(val_b), styles['TableCell']), 
            Paragraph(format_val(val_m), styles['TableCell'])
        ])
        
    t_comp = Table(comp_data, colWidths=[240, 110, 110])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 12))

    # Dynamic Note based on chosen metric
    fair_info = (selections or {}).get('fairness', {})
    metric_name = fair_info.get('specific_metric', '')

    if "Equalized Odds" in metric_name:
        note_text = "<b>Note:</b> An 'Equal Opportunity Difference' and 'Average Odds Difference' closer to 0 indicate a fairer model (balanced error rates)."
    else:
        # Default to Demographic Parity note
        note_text = "<b>Note:</b> A 'Statistical Parity Difference' closer to 0 and a 'Disparate Impact' closer to 1 indicate a fairer model."

    story.append(Paragraph(note_text, styles['Small']))
    story.append(PageBreak())

    # 5. Visualizations with Explanations
    if plots:
        story.append(Paragraph("4. Visualization Analysis", styles['Heading2']))
        story.append(Spacer(1, 12))
        
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
            if title == "Fairness-Utility Trade-off": continue
            if os.path.exists(img_path):
                story.append(Paragraph(title, styles['Heading3']))
                explanation = plot_explanations.get(title, "Visualization of dataset characteristics.")
                story.append(Paragraph(explanation, styles['Normal']))
                story.append(Spacer(1, 6))
                im = Image(img_path, width=400, height=300) 
                story.append(im)
                story.append(Spacer(1, 12))

    # 6. Recommendations
    story.append(Spacer(1, 18))
    story.append(Paragraph("5. Recommendations for Next Iterations", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    # Sub-section: Strategic Suggestions (Data-Driven)
    story.append(Paragraph("Strategic Suggestions (Data-Driven):", styles['Heading3']))
    if recommendations:
        for rec in recommendations:
            # Combine category and action into a classic bullet
            story.append(Paragraph(f"• <b>{rec.category}:</b> {rec.action}", styles['Normal']))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No specific recommendations triggered based on statistical thresholds.", styles['Normal']))

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
