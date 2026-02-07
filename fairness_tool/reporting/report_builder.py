from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os

def generate_pdf_report(filename, stats_before, metrics_before, metrics_after, plots=None, output_dir="outputs/reports", stats_after=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    filepath = os.path.join(output_dir, filename)
    doc = SimpleDocTemplate(filepath, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("Fairness Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Dataset Statistics Comparison Table
    story.append(Paragraph("Dataset Statistics Comparison", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    stat_comp_data = [['Statistic', 'Baseline', 'Mitigated']]
    all_stat_keys_set = set(stats_before.keys()) | set((stats_after or {}).keys())
    
    # Define preferred order
    preferred_order = [
        'Target Variable',
        'Features Selected',
        'Categorical Features',
        'Numerical Features',
        'Total Features',
        'Total Samples',
        'Missing Values'
    ]
    
    # Start with preferred keys that actually exist
    ordered_keys = [k for k in preferred_order if k in all_stat_keys_set]
    # Add remaining keys alphabetically
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
    
    # Comparison Table (Fairness & Performance)
    story.append(Paragraph("Fairness & Performance Comparison", styles['Heading2']))
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
    story.append(Spacer(1, 18))
    
    # Visualizations
    if plots:
        story.append(Paragraph("Visualizations", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        for title, img_path in plots.items():
            if os.path.exists(img_path):
                story.append(Paragraph(title, styles['Heading3']))
                story.append(Spacer(1, 6))
                
                # Resize image to fit page width (approx 6 inches = 432 pts)
                im = Image(img_path, width=400, height=300) 
                story.append(im)
                story.append(Spacer(1, 12))
            else:
                story.append(Paragraph(f"Image not found: {img_path}", styles['Normal']))

    doc.build(story)
    print(f"Report saved to {filepath}")