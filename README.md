# Fairness Toolkit for Tabular Datasets (CLI)

This is a command-line tool for analyzing and mitigating bias in tabular datasets.

## Setup

1.  **Install Dependencies:**
    Ensure you have Python 3.8+ installed. Run the following command to install the required libraries:

    ```bash
    pip install -r fairness_tool/requirements.txt
    ```

2.  **Run the Application:**
    Navigate to the project root and run:

    ```bash
    python fairness_tool/main.py
    ```

## Workflow

1.  **Load Dataset:** Provide the path to your `.csv` or `.xlsx` file.
2.  **EDA:** View basic statistics and plots (saved in `outputs/reports/assets`).
3.  **Preprocessing:** Handle missing values, outliers, and encoding interactively.
4.  **Fairness Setup:** Define sensitive attributes (e.g., Race, Gender) and privileged groups.
5.  **Baseline Evaluation:** Compute fairness and performance metrics on the original data.
6.  **Mitigation:** Apply bias mitigation techniques (e.g., Resampling).
7.  **Post-Evaluation:** Compare metrics after mitigation.
8.  **Output:** A PDF report and the improved dataset are saved in `outputs/`.

## Directory Structure

*   `fairness_tool/`: Source code.
*   `fairness_tool/outputs/`: Generated reports, plots, and datasets.
*   `fairness_tool/config/`: Configuration files.

## Recent Improvements

*   **Robust Evaluation:** Added 5-Fold Stratified Cross-Validation to the model training pipeline.
*   **Enhanced Metrics:** Included ROC AUC score and weighted F1/Precision/Recall for better handling of imbalanced datasets.
*   **Overfitting Check:** Improved overfitting detection using Cross-Validation accuracy instead of training accuracy.
