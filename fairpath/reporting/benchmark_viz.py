import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from fairpath.reporting.benchmark_analysis import BenchmarkAnalyzer

class BenchmarkVisualizer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self._load_data(file_path)
        self.output_dir = os.path.dirname(file_path)
        # Set style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Loads and validates the benchmark results file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel.")
            
        # Basic validation
        required_cols = ['Dataset', 'Model Type', 'Mitigation Technique', 'Accuracy']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        return df

    def filter_data(self, 
                   dataset: Optional[str] = None, 
                   model: Optional[str] = None, 
                   mitigation: Optional[str] = None) -> pd.DataFrame:
        """Filters the dataframe based on provided criteria."""
        df_filtered = self.df.copy()
        if dataset:
            df_filtered = df_filtered[df_filtered['Dataset'] == dataset]
        if model:
            df_filtered = df_filtered[df_filtered['Model Type'] == model]
        if mitigation:
            df_filtered = df_filtered[df_filtered['Mitigation Technique'] == mitigation]
        return df_filtered

    def plot_tradeoff(self, 
                     x_metric: str = 'Accuracy', 
                     y_metric: str = 'Statistical Parity Difference', 
                     dataset: Optional[str] = None,
                     save_path: Optional[str] = None):
        """Generates a scatter plot for Fairness-Utility Trade-off."""
        df_plot = self.filter_data(dataset=dataset)
        
        if x_metric not in df_plot.columns or y_metric not in df_plot.columns:
            print(f"Metrics {x_metric} or {y_metric} not found in data. Skipping tradeoff plot.")
            return

        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        ax = sns.scatterplot(
            data=df_plot,
            x=x_metric,
            y=y_metric,
            hue='Model Type',
            style='Mitigation Technique',
            s=100,
            alpha=0.7
        )
        
        plt.title(f'{y_metric} vs {x_metric} Trade-off' + (f' ({dataset})' if dataset else ''))
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)  # Zero line for fairness
        
        # Check if legend handles exist before creating legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_grouped_metrics(self, 
                            metrics: List[str], 
                            group_by: str = 'Model Type',
                            dataset: Optional[str] = None,
                            save_path: Optional[str] = None):
        """Generates grouped bar charts for specified metrics."""
        df_plot = self.filter_data(dataset=dataset)
        
        # Melt dataframe for easier plotting with seaborn
        available_metrics = [m for m in metrics if m in df_plot.columns]
        if not available_metrics:
            print("No requested metrics found in data.")
            return
            
        df_melted = df_plot.melt(
            id_vars=[group_by, 'Mitigation Technique'], 
            value_vars=available_metrics, 
            var_name='Metric', 
            value_name='Value'
        )
        
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df_melted,
            x=group_by,
            y='Value',
            hue='Metric',
            errorbar=None  # Remove error bars for cleaner view, or use 'sd' for standard deviation
        )
        
        plt.title(f'Performance Metrics by {group_by}' + (f' ({dataset})' if dataset else ''))
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_metric_distribution(self, 
                               metric: str, 
                               group_by: str = 'Model Type', 
                               dataset: Optional[str] = None,
                               save_path: Optional[str] = None):
        """Generates box plots to show distribution of a metric across runs."""
        df_plot = self.filter_data(dataset=dataset)
        
        if metric not in df_plot.columns:
            print(f"Metric {metric} not found.")
            return

        # Check for empty data or all-NaN metric column to prevent seaborn errors
        if df_plot.empty or df_plot[metric].dropna().empty:
            print(f"No valid data available for metric '{metric}' in dataset '{dataset}'. Skipping distribution plot.")
            return

        plt.figure(figsize=(10, 6))
        try:
            sns.boxplot(
                data=df_plot,
                x=group_by,
                y=metric,
                hue='Mitigation Technique',
                orient='v'
            )
            
            plt.title(f'Distribution of {metric}' + (f' ({dataset})' if dataset else ''))
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        except Exception as e:
            print(f"Error plotting distribution for {metric}: {e}")
            plt.close()
            
    def plot_heatmap(self,
                    metric: str,
                    row_factor: str = 'Model Type',
                    col_factor: str = 'Mitigation Technique',
                    dataset: Optional[str] = None,
                    save_path: Optional[str] = None):
        """Generates a heatmap of a metric averaged over runs."""
        df_plot = self.filter_data(dataset=dataset)
        
        if metric not in df_plot.columns:
            print(f"Metric {metric} not found.")
            return

        # Aggregate data
        try:
            pivot_table = df_plot.pivot_table(
                values=metric, 
                index=row_factor, 
                columns=col_factor, 
                aggfunc='mean'
            )
            
            # Check for empty pivot table to avoid seaborn errors
            if pivot_table.empty or pivot_table.dropna(how='all').empty:
                print(f"No valid data available for heatmap of '{metric}' in dataset '{dataset}'. Skipping.")
                return

            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={'label': metric})
            
            plt.title(f'Average {metric} Heatmap' + (f' ({dataset})' if dataset else ''))
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        except Exception as e:
            print(f"Error plotting heatmap for {metric}: {e}")
            plt.close()

    def plot_pareto_frontier(self,
                           x_metric: str = 'Accuracy',
                           y_metric: str = 'Statistical Parity Difference',
                           dataset: Optional[str] = None,
                           save_path: Optional[str] = None):
        """Highlights the Pareto frontier (optimal trade-offs)."""
        df_plot = self.filter_data(dataset=dataset)
        if x_metric not in df_plot.columns or y_metric not in df_plot.columns:
            return

        # Assuming we want to maximize Accuracy and minimize absolute Fairness Diff
        # Adjust directionality as needed. Here we assume:
        # X: Higher is better (e.g., Accuracy)
        # Y: Closer to 0 is better (e.g., Diff). We will transform Y to abs() and then minimize it.
        
        # Prepare data for calculation
        df_plot['abs_y'] = df_plot[y_metric].abs()
        
        # Simple Pareto calculation: A point is on the frontier if no other point has (better X AND better Y)
        # Better X: > x
        # Better Y: < abs_y
        
        frontier = []
        for i, row in df_plot.iterrows():
            is_dominated = False
            for j, other in df_plot.iterrows():
                if i == j: continue
                if (other[x_metric] >= row[x_metric] and other['abs_y'] <= row['abs_y']) and \
                   (other[x_metric] > row[x_metric] or other['abs_y'] < row['abs_y']):
                    is_dominated = True
                    break
            if not is_dominated:
                frontier.append(row)
        
        df_frontier = pd.DataFrame(frontier)
        
        plt.figure(figsize=(10, 6))
        # Plot all points (background)
        plt.scatter(
            df_plot[x_metric],
            df_plot[y_metric],
            color='gray',
            alpha=0.5,
            label='All Configurations'
        )
        
        # Plot frontier points
        if not df_frontier.empty:
            sns.scatterplot(
                data=df_frontier,
                x=x_metric,
                y=y_metric,
                hue='Model Type',
                style='Mitigation Technique',
                s=150,
                edgecolor='black',
                zorder=10
            )
            
            # Draw line connecting frontier (sorted by X)
            df_sorted = df_frontier.sort_values(by=x_metric)
            plt.plot(df_sorted[x_metric], df_sorted[y_metric], 'r--', alpha=0.5, label='Pareto Frontier')

        plt.title(f'Pareto Frontier: {y_metric} vs {x_metric}' + (f' ({dataset})' if dataset else ''))
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        # Move legend outside
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def generate_full_report(self, output_dir: Optional[str] = None):
        """Generates a standard set of plots for all datasets found."""
        target_dir = output_dir if output_dir else os.path.join(self.output_dir, "viz_report")
        os.makedirs(target_dir, exist_ok=True)
        
        datasets = self.df['Dataset'].unique()
        
        print(f"Generating report in {target_dir}...")
        
        for ds in datasets:
            clean_ds_name = "".join(x for x in ds if x.isalnum() or x in " _-").strip().replace(" ", "_")
            ds_dir = os.path.join(target_dir, clean_ds_name)
            os.makedirs(ds_dir, exist_ok=True)
            
            # 1. Trade-off
            self.plot_tradeoff(
                dataset=ds, 
                save_path=os.path.join(ds_dir, "tradeoff_acc_parity.png")
            )
            self.plot_tradeoff(
                dataset=ds,
                y_metric='Equal Opportunity Difference',
                save_path=os.path.join(ds_dir, "tradeoff_acc_eqopp.png")
            )
            
            # 2. Grouped Bar Charts
            self.plot_grouped_metrics(
                metrics=['Accuracy', 'F1', 'Recall'], 
                dataset=ds, 
                save_path=os.path.join(ds_dir, "metrics_performance.png")
            )
            
            # 3. Distributions
            self.plot_metric_distribution(
                metric='Accuracy', 
                dataset=ds, 
                save_path=os.path.join(ds_dir, "dist_accuracy.png")
            )
            self.plot_metric_distribution(
                metric='Statistical Parity Difference', 
                dataset=ds, 
                save_path=os.path.join(ds_dir, "dist_parity.png")
            )
            
            # 4. Heatmaps
            self.plot_heatmap(
                metric='Accuracy', 
                dataset=ds, 
                save_path=os.path.join(ds_dir, "heatmap_accuracy.png")
            )
            self.plot_heatmap(
                metric='Statistical Parity Difference', 
                dataset=ds, 
                save_path=os.path.join(ds_dir, "heatmap_parity.png")
            )
            
            # 5. Pareto
            self.plot_pareto_frontier(
                dataset=ds,
                save_path=os.path.join(ds_dir, "pareto_frontier.png")
            )
        
        # Analyze outliers
        analyzer = BenchmarkAnalyzer(self.df)
        outliers_data = analyzer.detect_high_performance_outliers(metric="Accuracy") # Using Accuracy as default for now
        
        self.generate_pdf_summary(target_dir, datasets, outliers_data)
        print(f"Report generation complete. Check {target_dir}")

    def generate_pdf_summary(self, base_dir: str, datasets: List[str], outliers_data: Dict[str, Tuple[pd.DataFrame, bool]] = {}):
        """Generates a PDF summary report with all plots and outlier tables."""
        filepath = os.path.join(base_dir, "benchmark_summary_report.pdf")
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("Benchmark Visualization Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("This report contains visual summaries of the fairness benchmark runs.", styles['Normal']))
        story.append(Spacer(1, 24))
        
        for ds in datasets:
            clean_ds_name = "".join(x for x in ds if x.isalnum() or x in " _-").strip().replace(" ", "_")
            ds_dir = os.path.join(base_dir, clean_ds_name)
            
            story.append(Paragraph(f"Dataset: {ds}", styles['Heading1']))
            story.append(Spacer(1, 12))
            
            # List of expected plots and their descriptions
            plots = [
                ("tradeoff_acc_parity.png", "Fairness-Utility Trade-off (Accuracy vs. Parity)", 
                 "Scatter plot showing the relationship between Accuracy and Statistical Parity Difference. Points closer to (1, 0) are ideal."),
                ("tradeoff_acc_eqopp.png", "Fairness-Utility Trade-off (Accuracy vs. Equal Opportunity)",
                 "Scatter plot showing the relationship between Accuracy and Equal Opportunity Difference."),
                ("pareto_frontier.png", "Pareto Frontier",
                 "Highlights the optimal trade-off points where no other configuration offers both better performance and better fairness."),
                ("metrics_performance.png", "Performance Metrics Comparison",
                 "Bar charts comparing key performance metrics (Accuracy, F1, Recall) across different models."),
                ("dist_accuracy.png", "Accuracy Distribution",
                 "Box plot showing the spread of accuracy scores across multiple runs."),
                ("dist_parity.png", "Fairness Metric Distribution",
                 "Box plot showing the spread of fairness scores across multiple runs."),
                ("heatmap_accuracy.png", "Accuracy Heatmap",
                 "Heatmap of average accuracy for each Model-Mitigation combination."),
                ("heatmap_parity.png", "Fairness Heatmap",
                 "Heatmap of average fairness score for each Model-Mitigation combination.")
            ]
            
            for filename, title, desc in plots:
                img_path = os.path.join(ds_dir, filename)
                if os.path.exists(img_path):
                    story.append(Paragraph(title, styles['Heading2']))
                    story.append(Spacer(1, 6))
                    story.append(Paragraph(desc, styles['Normal']))
                    story.append(Spacer(1, 12))
                    
                    # Resize image to fit page width roughly
                    try:
                        img = Image(img_path, width=450, height=300)
                        story.append(img)
                        story.append(Spacer(1, 12))
                    except Exception as e:
                        print(f"Warning: Could not add image {filename} to PDF: {e}")
            
            # --- High-Performance Configurations Section ---
            if ds in outliers_data:
                df_out, is_outlier = outliers_data[ds]
                
                if not df_out.empty:
                    story.append(PageBreak())
                    
                    if is_outlier:
                        section_title = f"High-Performance Outliers: {ds}"
                        desc_text = "The following configurations achieved unusually high accuracy compared to the rest of the runs (IQR outlier detection)."
                    else:
                        section_title = f"Top Performing Configurations: {ds}"
                        desc_text = "No statistical outliers were found. The following table shows the top performing configurations for this dataset."
                    
                    story.append(Paragraph(section_title, styles['Heading2']))
                    story.append(Spacer(1, 6))
                    story.append(Paragraph(desc_text, styles['Normal']))
                    story.append(Spacer(1, 12))
                    
                    # Prepare data for table
                    analyzer = BenchmarkAnalyzer(self.df)
                    df_formatted = analyzer.format_outlier_table(df_out)
                    
                    # Limit columns for PDF readability - Pick key ones
                    key_cols = ['Model Type', 'Mitigation Technique', 'Mitigation Detail', 'Accuracy', 'Statistical Parity Difference']
                    # Add 'F1' if exists
                    if 'F1' in df_formatted.columns: key_cols.insert(4, 'F1')
                    
                    # Ensure we have data to show
                    df_display = df_formatted[key_cols].head(10)
                    
                    # Convert DataFrame to list of lists for ReportLab
                    # Use simpler values for display (round numbers)
                    table_data = [df_display.columns.tolist()]
                    for row in df_display.values:
                        formatted_row = []
                        for item in row:
                            if isinstance(item, float):
                                formatted_row.append(f"{item:.4f}")
                            else:
                                formatted_row.append(str(item))
                        table_data.append(formatted_row)
                    
                    # Create Table
                    t = Table(table_data)
                    
                    # Style the table
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('FONTSIZE', (0, 0), (-1, -1), 8), # Smaller font
                    ]))
                    
                    story.append(t)
                    story.append(Spacer(1, 12))
                    
            story.append(PageBreak())
            
        try:
            doc.build(story)
            print(f"PDF Summary saved to {filepath}")
        except Exception as e:
            print(f"Error building PDF: {e}")
