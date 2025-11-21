# — Interpretable AI: SHAP & LIME Analysis of a Gradient Boosting Model for Credit Risk Assessment

## 📌 Project Overview

This project demonstrates how to build a fully **interpretable credit risk prediction system** using a **Gradient Boosting Machine (GBM)**.
The objective is to predict **loan default likelihood** using the **German Credit dataset (or similar credit dataset)** and apply **post-hoc explainability techniques** such as:

* **SHAP (SHapley Additive exPlanations)** → global interpretability
* **LIME (Local Interpretable Model-agnostic Explanations)** → local instance-level interpretability

The project highlights not only predictive performance but also **model transparency, fairness, and business impact**, simulating a real-world loan underwriting scenario where ethical and interpretable AI is mandatory.

---

# 📂 Dataset

The dataset contains credit applicant information such as:

* Demographics
* Loan duration & amount
* Account history
* Employment information
* Purpose of loan
* Default status (target variable: 0 = no default, 1 = default)

---

# 🧠 Project Workflow (Aligned With “Tasks to Complete”)

### **1️⃣ Data Preprocessing**

* Loaded the credit dataset from CSV
* Addressed missing values
* Encoded categorical variables using **One-Hot Encoding**
* Scaled numeric variables using **StandardScaler**
* Split data into **training (80%)** and **test (20%)** sets
* Optionally applied **SMOTE** for handling class imbalance

---

### **2️⃣ Model Training & Hyperparameter Optimization**

* Trained a **Gradient Boosting Classifier (XGBoost)**
* Tuned major hyperparameters such as:

  * `n_estimators`
  * `learning_rate`
  * `max_depth`
  * `subsample`
  * `colsample_bytree`
* Used a validation set with **early stopping**
* Optimized for **AUC**, **Recall**, and minimizing false negatives
  (critical for credit risk use-cases)

---

### **3️⃣ Global Explainability Using SHAP**

Generated and analyzed:

* **SHAP Summary Plot** (overall drivers of default)
* **SHAP Feature Importance Rankings**
* **SHAP Dependence Plots**
* Identified the **Top 5 most influential features** affecting default probability

These global explanations reveal the key risk factors for loan default.

---

### **4️⃣ Local Explainability Using LIME**

Selected three representative test cases:

1. **Clear Positive Case** (customer who truly defaulted)
2. **Clear Negative Case** (customer who did NOT default)
3. **Borderline Case** (model uncertainty ~0.5 probability)

For each case, LIME generated:

* Local explanation plots
* Top contributing features
* Comparison against SHAP for consistency

---

### **5️⃣ Final Business Insights + Actionable Recommendations**

Based on SHAP & LIME results:

* Summarized model performance
* Explained global + local interpretability findings
* Proposed **2–3 business rules** such as:

  * Automatically flag applicants with high-risk SHAP features
  * Require manual review for borderline LIME cases
  * Adjust policy thresholds for high-risk demographic/financial signals

---

# 🎯 Expected Deliverables (Completed)

### **1. Complete, runnable Python code**

Included in

```
interpretable_gbm_shap_lime.py
```

This script runs end-to-end:

* Data loading
* Preprocessing
* Training
* SHAP & LIME analysis
* Evaluation
* Output saving

---

### **2. Model Performance Report**

Generated metrics include:

* **AUC**
* **Precision**
* **Recall**
* **Classification Report**
* **ROC Curve**

Saved as:

```
outputs/metrics.json
outputs/roc_curve.png
outputs/report.txt
```

---

### **3. Global Feature Importance (SHAP) Analysis**

Delivered outputs:

```
outputs/shap_summary.png
outputs/shap_dependence_<feature>.png
```

With top 5 drivers fully documented.

---

### **4. Local Case Study Explanations (LIME)**

Three local explanations saved as:

```
outputs/lime_positive_high_prob.png
outputs/lime_negative_low_prob.png
outputs/lime_borderline.png
```

Also summarised in `report.txt`.

---

### **5. Actionable Business Recommendations**

Presented in the final section of the report, examples include:

* Introduce risk-based loan approval tiers
* Add manual underwriting for borderline cases
* Use SHAP top drivers to refine credit score policy

---

# 🗂 Output Directory Structure

```
outputs/
│── model.pkl
│── preprocessor.pkl
│── metrics.json
│── report.txt
│── roc_curve.png
│── shap_summary.png
│── shap_dependence_<feature>.png
│── lime_positive_high_prob.png
│── lime_negative_low_prob.png
│── lime_borderline.png
```

---

# 🚀 How to Run the Script

### **Command Line**

```
python interpretable_gbm_shap_lime.py --data "<path_to_credit_csv>" --target_col default
```

Example:

```
python interpretable_gbm_shap_lime.py --data "E:\german_credit.csv" --target_col default
```

---

# 💡 Technologies Used

* Python
* XGBoost
* SHAP
* LIME
* Scikit-learn
* Pandas
* Matplotlib
* Imbalanced-learn

---

# 🤝 Author

**Swetha G V**
Project: Interpretable AI — SHAP & LIME for Credit Risk Assessment
Created as part of a structured AI/ML learning program.


Below is a **clean, simple, detailed, and easy-to-understand version of the Expected Deliverables** for your project.
You can copy-paste this directly into your submission.

---

# **Expected Deliverables**

## **1. Complete, runnable Python code implementation**

The project should include a full Python script that performs all steps from start to finish. The code must clearly show:

* **Loading and preprocessing the credit dataset:**
  Handling missing values, encoding categorical variables, normalizing/standardizing features, and splitting data into train/test sets.

* **Training the Gradient Boosting Model:**
  Using XGBoost or LightGBM, performing hyperparameter tuning (e.g., GridSearchCV or RandomizedSearchCV), and optimizing mainly for **AUC** and **Recall** (because catching defaulters is important).

* **SHAP Analysis:**
  Generating global explanations such as:

  * Overall feature importance
  * SHAP summary plots
  * SHAP dependence plots for major features

* **LIME Analysis:**
  Using 3 selected test cases (1 positive, 1 negative, 1 borderline) and generating local explanations.

* **Clear comments** throughout the code explaining what each section does.

The Python script should be **directly runnable** without errors when the dataset is provided.

---

## **2. Text-based report section with model performance metrics**

This section should present all final evaluation metrics after testing the model. Your report should include:

* **AUC (Area Under ROC Curve):**
  Measures the model’s ability to correctly classify defaulters vs. non-defaulters.

* **Precision:**
  Out of all predicted defaulters, how many were actually defaulters.

* **Recall (important metric for credit risk):**
  Out of all actual defaulters, how many the model successfully caught.

* **Confusion Matrix Interpretation:**
  Short explanation of True Positives, True Negatives, False Positives, and False Negatives.

Your text should clearly explain what these metrics mean in a business context—for example, higher recall reduces the chance of giving loans to risky customers.

---

## **3. Written comparison of global feature importances: GBM model vs. SHAP**

This section must explain:

* **What features the GBM model internally considers most important**
  (based on built-in feature importance scores like gain, split, cover)

* **What SHAP identifies as the top drivers of loan default risk**

* **How SHAP adds deeper interpretability**
  such as direction of influence (positive = increases default risk, negative = reduces risk)

You should compare and explain:

* Where both methods agree (e.g., “Age” and “Credit History” were top features in both).
* Where they differ and why (e.g., GBM may highlight frequency of splits, while SHAP reveals real impact on prediction probability).

This comparison shows understanding of **model structure vs. post-hoc explanation techniques**.

---

## **4. Detailed summary of SHAP and LIME results for 3 selected instances**

Choose:

* **1 clear positive (high default risk)**
* **1 clear negative (low default risk)**
* **1 borderline or uncertain case**

For each instance:

### Provide:

* **SHAP force plot explanation:**
  Which features pushed the prediction toward default or non-default.

* **LIME explanation:**
  A local linear approximation showing the top features affecting that prediction.

### Then explain:

* How SHAP and LIME **agree** (e.g., both highlight high loan amount as risky)
* How they **differ** (e.g., LIME may give higher weight to employment status, while SHAP focuses more on credit duration)
* Why such differences occur
  (LIME is local & linear; SHAP is game-theoretic with consistent attribution)

This item should read like three small case studies with clear reasoning.

---

## **5. 2–3 actionable business recommendations based on model insights**

These recommendations should be practical, simple, and connected to the model’s findings.

Examples:

1. **Strengthen loan approval rules for customers with poor credit history or unstable employment**, since both SHAP and LIME highlight these as major drivers of default risk.

2. **Introduce a tiered interest rate system** where customers with moderate risk scores receive a higher but manageable interest rate instead of outright rejection.

3. **Use the model’s explanations during manual review**, helping loan officers understand why a customer was flagged as risky and allowing more transparent decisions.

The recommendations must be clearly tied to the features and patterns revealed by the model.

--



