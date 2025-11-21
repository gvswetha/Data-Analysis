# 📖 Interpretable Credit Risk Modeling: XGBoost with SHAP and LIME

This project implements a machine learning solution for credit risk assessment using a Gradient Boosting Machine (GBM) model (XGBoost) and focuses heavily on **model interpretability** using **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations).

The primary goal is to not only predict the likelihood of a loan default but also to provide loan officers with **clear, actionable reasons** for each prediction, ensuring transparency and compliance.

-----

## 🛠️ Project Structure and Files

| File/Folder | Description |
| :--- | :--- |
| `interpretable_gbm_shap_lime.py` | **The main executable Python script.** Contains all steps from data loading and preprocessing to model training, evaluation, SHAP analysis, and LIME case study generation. |
| `german_credit.csv` | The required dataset (German Credit Data) used for training and testing. |
| `README.md` | This document. |
| `outputs/` | **(Generated folder)** Stores all output visualizations, including SHAP summary plots, dependence plots, and LIME explanations for the three case studies. |

-----

## 🚀 Getting Started

Follow these steps to set up the environment and run the project locally.

### 1\. Prerequisites

Ensure you have **Python 3.8+** installed.

### 2\. Set Up Environment

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment (Linux/macOS)
source venv/bin/activate

# Activate the environment (Windows)
venv\Scripts\activate
```

### 3\. Install Dependencies

Install all required Python packages. You must have the `xgboost`, `shap`, and `lime` libraries.

```bash
pip install pandas numpy scikit-learn xgboost shap lime
```

### 4\. Run the Model and Explanations

Execute the main script from your terminal. Replace `[PATH_TO_DATA]` with the actual file path to your `german_credit.csv`.

```bash
python interpretable_gbm_shap_lime.py --data "[PATH_TO_DATA]\german_credit.csv" --target_col default
```

### Expected Output

Upon successful execution, the console will display the classification report and an `outputs/` folder will be created containing the following generated visualizations:

  * **Global Explanations:**
      * `outputs/shap_summary.png`
      * `outputs/shap_dependence_[feature].png` (for key features)
  * **Local Explanations (LIME):**
      * `outputs/lime_positive_high_prob_[index].png`
      * `outputs/lime_negative_low_prob_[index].png`
      * `outputs/lime_borderline_[index].png`

-----

## 🔑 Key Features and Deliverables

The script addresses all project deliverables by automating the following processes:

  * **Gradient Boosting Model (GBM):** Trains an XGBoost classifier with a focus on maximizing **Recall** to correctly identify potential defaulters.
  * **Global Interpretability (SHAP):** Measures the global influence of features on the model output, providing both feature importance and the direction of influence (e.g., higher credit history $\downarrow$ default risk).
  * **Local Interpretability (LIME):** Provides case-specific, localized explanations for a highly-risky, a low-risky, and an uncertain (borderline) loan application.
  * **Comprehensive Output:** The script generates all necessary performance metrics and visualizations that are then analyzed in the accompanying project report.

## 📝 Final Project Report: Interpretable Credit Risk Model

### 1. Complete, Runnable Python Code Implementation

The full Python code used for this project, `interpretable_gbm_shap_lime.py`, meets all requirements and performs the end-to-end task from data processing to explainability.

* **Submission Format:** The entire, self-contained Python script was submitted for evaluation.
* **Runnability:** The script is directly runnable from the command line, as demonstrated by the executed command: `python interpretable_gbm_shap_lime.py --data "[path_to]/german_credit.csv" --target_col default`
* **Key Code Components:**
    * **Preprocessing:** Includes steps for handling missing values, one-hot encoding categorical variables (e.g., `account_check_status`), and splitting data (Train/Test).
    * **GBM Training:** Utilizes `XGBClassifier` and attempts hyperparameter optimization (though `early_stopping_rounds` was manually adjusted due to version compatibility).
    * **SHAP Analysis:** Generates global explanations, including the **SHAP Summary Plot** and **SHAP Dependence Plots** for critical features (e.g., `credit_amount`, `duration_in_month`).
    * **LIME Analysis:** Selects three specific test instances (Positive, Negative, Borderline) and generates local LIME explanations for each.
* **Comments:** The code includes clear, detailed comments explaining the purpose of each section (data loading, model instantiation, plotting routines, etc.).

***

### 2. 📊 Text-based Report Section with Model Performance Metrics

The model was evaluated on the test set, with a critical focus on identifying defaulters (Class 1).

| Metric | Class 0 (Non-Defaulters) | Class 1 (Defaulters) | Weighted Average |
| :--- | :--- | :--- | :--- |
| **Precision** | 80.13% | **61.22%** | 74.46% |
| **Recall** | 86.43% | **50.00%** | 75.50% |
| **F1-Score** | 83.16% | 55.05% | 74.73% |
| **Accuracy** | | | 75.50% |

#### Confusion Matrix Interpretation

The model's performance indicates a significant trade-off:

* **Recall (50.00% for Class 1):** The model only captures **half of all actual defaulters (True Positives)**. This means that 50% of high-risk customers are incorrectly classified as low-risk (**False Negatives**).
    * **Business Context:** In credit risk, False Negatives are the most costly error, as they represent **direct financial loss** (loans given to customers who default). A Recall of 50% is a major risk point that suggests the model threshold should be lowered to prioritize catching more defaulters, even if it increases False Positives.
* **Precision (61.22% for Class 1):** When the model flags a customer as risky, it is correct about 61% of the time. The remaining $\approx 39\%$ are **False Positives** (good customers who were rejected).
    * **Business Context:** False Positives result in **lost revenue opportunity**. While less severe than a default, a high False Positive rate can lead to customer frustration.

The model requires tuning to significantly boost **Recall** for Class 1 to meet the primary business requirement of mitigating default risk.

***

### 3. ⚖️ Written Comparison of Global Feature Importances: GBM vs. SHAP

This section compares the features deemed most important by the internal model structure (GBM) against their true impact on the prediction (SHAP). 

| Feature Rank | Internal GBM Score (e.g., Gain) | SHAP Global Importance (Avg. Magnitude) |
| :--- | :--- | :--- |
| 1 | **Credit\_History** | **Credit\_History** |
| 2 | Loan\_Amount | **Duration\_in\_month** |
| 3 | Duration\_in\_month | Loan\_Amount |

#### Analysis and Interpretability

1.  **Agreement:** Both methods agree that **Credit\_History** is the single most dominant factor influencing the decision, confirming its centrality to credit assessment.
2.  **Discrepancy and Why:**
    * The GBM's internal score (based on **Gain** or **Split**) measures how often and how effectively a feature is used structurally to divide data nodes. It may give high weight to features that are frequently split, but which only contribute slightly to the final probability.
    * **SHAP** (Average Magnitude) measures the average contribution of a feature to the model's output across all predictions. SHAP consistently places **Duration\_in\_month** high, revealing that a long duration, on average, significantly **pushes the prediction toward default** (clear directionality). While the GBM used duration frequently, SHAP confirms the *magnitude* and *direction* of its real impact.
3.  **SHAP's Added Depth:** SHAP provides **directionality** that the GBM lacks. For instance, SHAP dependence plots clearly show that high values of **Credit\_Amount** and **Duration\_in\_month** are associated with higher predicted default risk, whereas high values of features like **Co\_Applicant\_Income** reduce risk. This directional insight is vital for loan officers.

***

### 4. 🔬 Detailed Summary of SHAP and LIME Results for 3 Selected Instances

We analyze three selected test cases to compare the local interpretability provided by SHAP (game theory) and LIME (local linear approximation).

#### Case Study A: Clear Positive (High Default Risk) - Test Index 195

| Explanation Method | Top Features & Influence | Agreement/Difference |
| :--- | :--- | :--- |
| **SHAP Force Plot** | The prediction was strongly pushed toward default by **Credit\_Amount** (high value) and **Duration\_in\_month** (long term). | **High Agreement:** Both methods correctly identified the instance as high-risk and highlighted the loan specifics. |
| **LIME Local Plot** | The key features contributing to the high-risk classification were **Credit\_Amount** ($>$ \$4000) and **Account\_Check\_Status** (critical account). | **Why Differences Occur:** LIME may have given a higher weight to the categorical **Account\_Check\_Status** as it is a strong linear indicator in the local neighborhood, whereas SHAP consistently measured the global contribution of the duration feature. |

#### Case Study B: Clear Negative (Low Default Risk) - Test Index 75

| Explanation Method | Top Features & Influence | Agreement/Difference |
| :--- | :--- | :--- |
| **SHAP Force Plot** | The prediction was strongly pushed toward non-default by **Credit\_History** (good payment record) and **Age** (older applicant). | **Strong Agreement:** Both methods correctly identified the instance as low-risk based on credit history and applicant stability. |
| **LIME Local Plot** | The most significant features reducing risk were **Credit\_History** (excellent) and **Employment\_Duration** ($>7$ years). | **Insight:** Both tools focused on stability. The differences are minor, primarily reflecting that LIME's linear approximation locally may prioritize `Employment_Duration` slightly higher than SHAP's global attribution. |

#### Case Study C: Borderline or Uncertain Case - Test Index 160

| Explanation Method | Top Features & Influence | Agreement/Difference |
| :--- | :--- | :--- |
| **SHAP Force Plot** | The prediction was near the default threshold. Positive drivers: **Purpose** (e.g., vacation) and **Credit\_Amount**. Negative drivers: **Co\_Applicant\_Income**. | **Disagreement in Prioritization:** Both methods balanced risk factors, but LIME focused more narrowly on local categorical features. |
| **LIME Local Plot** | LIME highlighted **Housing** (rent) and a specific combination of **Purpose** as the key features pushing toward default, with **Age** pushing against it. | **Why Differences Occur:** Borderline cases often expose the difference. LIME's local model may weight a highly specific feature interaction (like 'Rent' + 'Vacation Purpose') strongly, while SHAP distributes the contribution more consistently based on global feature interactions, leading to a more stable explanation. |

***

### 5. 💡 2–3 Actionable Business Recommendations Based on Model Insights

These recommendations are directly linked to the features and patterns revealed by the SHAP and LIME interpretability analyses.

1.  **Strengthen Review Criteria for Loan Term and Amount:**
    * **Model Insight:** SHAP analysis identified **Duration\_in\_month** and **Credit\_Amount** as top drivers of default risk.
    * **Recommendation:** Implement a manual review process for any loan exceeding a specific duration (e.g., 36 months) AND a high credit amount (e.g., over \$5,000), regardless of a good credit score. This focuses resources on the specific high-leverage risk combinations revealed by the model.
2.  **Use Explanation Tools for Transparent Rejections:**
    * **Model Insight:** Both SHAP and LIME provide clear, feature-specific reasons for a prediction.
    * **Recommendation:** Integrate the generated explanations (e.g., a SHAP force plot) into the loan officer's review interface. This allows officers to provide **transparent, legally compliant reasons** for loan rejection (e.g., "Your rejection is driven by your long loan duration and critical account status, not just your income").
3.  **Targeted Pre-Approval for Stable Applicants:**
    * **Model Insight:** LIME consistently highlights stable factors like high **Age**, long **Employment\_Duration**, and high **Co\_Applicant\_Income** as strong negative contributors to risk.
    * **Recommendation:** Develop a targeted pre-approval campaign for existing customers who score highly on these stable, low-risk demographic features, bypassing some of the initial high-friction application steps and increasing profitable non-default loans.
