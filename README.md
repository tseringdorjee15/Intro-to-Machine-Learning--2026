# 1. Machine Learning Pipeline for Hospital Length-of-Stay Classification
### Raw NYC SPARCS "Streaming Ingestion" to "Multi-Architecture Model Selection"

**By:** Tsering Dorjee  
**Institution:** LaGuardia Community College, NYC  
**Dataset:** 2024 NYS SPARCS Inpatient Discharges (2.2M+ Rows) from health.data.ny.gov

---

## Project Overview
This project builds a machine learning pipeline to predict a patient's **Length of Stay (LoS)** category right at the time of hospital admission. 

We transformed raw stay durations into a binary target:
* **Normal Stay:** Less than 6 days (70.81% of data)
* **Prolonged Stay (PLoS):** 6 days or more (29.19% of data based on the 75th percentile)

---

## Data Strategy & Leakage Prevention
* **Leakage Controls:** Removed columns populated *after* discharge (`total_charges`, `total_costs`, `patient_disposition`) so the model only uses data known at admission. Dropped `id` and `version` columns.
* **Newborn Isolation:** Leveraged missing values in `birth_weight` to engineer a high-signal binary flag (`is_newborn`).
* **Feature Encoding:** 
  * **Logistic Regression:** One-Hot Encoded (expanded to 118 columns).
  * **Tree Ensembles:** Used **Target Encoding** to compress massive string codes into **17 memory-efficient columns** to prevent computer RAM crashes.

---

## Model Performance Comparison
Custom classification thresholds were used to maximize **Recall** (catching actual long-stay patients) to prioritize hospital safety.

| Metric | Logistic Regression (Baseline) | Random Forest (Ensemble) | Gradient Boosting (Boosting) |
| :--- | :---: | :---: | :---: |
| **Encoding Applied** | One-Hot Encoding | Target Encoding | Target Encoding |
| **Decision Threshold** | 50% (Default) | **35% (Optimized)** | **30% (Optimized)** |
| **Global Accuracy** | **80.61%** | 78.95% | 77.34% |
| **ROC-AUC Score** | 0.8199 | **0.8626** | 0.8621 |
| **Prolonged Stay Recall** | 0.56 | 0.75 | **0.78** |
| **Prolonged Stay F1-Score**| 0.63 | **0.67** | **0.67** |

## Final Confusion Matrices
* **Logistic Regression:** TN: 12,815 | TP: 3,306 | **FN: 2,550** | FP: 1,329
* **Random Forest:** TN: 11,432 | TP: 4,358 | **FN: 1,480** | FP: 2,730
* **Gradient Boosting:** TN: 10,901 | TP: 4,568 | **FN: 1,270** | FP: 3,261


## Conclusions

### A. Most Important Features
Evaluating feature importances proved that **clinical metrics matter more than administrative ones**. The top drivers of stay duration are:
1. `ccsr_procedure_code` (Primary surgical intervention)
2. `ccsr_diagnosis_code` (Admitting medical condition)
3. `zip_code` (Captures socioeconomic/regional variation)
4. `apr_severity_of_illness_code` (Patient sickness levels)

### B. Final Deployment Choice
The **Random Forest Classifier** was selected for final deployment. It achieved the highest overall discriminative performance (**ROC-AUC: 0.8626**). By lowering the threshold to 35%, it successfully flagged **1,052 prolonged-stay patients** that the standard baseline completely missed, while maintaining a much cleaner false-alarm profile than Gradient Boosting.

==================================================================================================================================

# 2. Titanic Survival - Classification Analysis

This project fulfills Assignment 2 for Intro to Machine Learning at LaGuardia Community College.

## Project Overview
- **Dataset:** Titanic Survival data.
- **Objective:** Train and evaluate a binary classification model to predict passenger survival.
- **Model Framework:** Built using a Logistic Regression model to establish an interpretive statistical baseline using passenger feature weights.
- **Evaluation Metrics:** Evaluated using a Classification Report, a Confusion Matrix Heatmap, and a Receiver Operating Characteristic (ROC) Curve achieving an AUC score of 0.80.

*Each code cell inside the notebook is annotated with explanations detailing the data pipeline and performance comparisons.*

====================================================================================================================================

# 3. Boston Housing - Linear Regression Analysis

This project fulfills Assignment 1 for Intro to Machine Learning at LaGuardia Community College. 

## Project Overview
- **Dataset:** Boston Housing data.
- **Objective:** Compare two versions of a Linear Regression model.
- **Model 1:** Built using all 13 features to capture complex market dynamics.
- **Model 2:** Built using a subset of features (Room Count and Lower-Class Status) for a clean, highly interpretable alternative.

*Each code cell inside the notebook is annotated with explanations detailing the data pipeline and performance comparisons.*

