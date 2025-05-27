<p align="center">
  <img src="school_image.png" alt="School illustration" width="400"/>
</p>

# Student Intervention System – Machine Learning Project

**Author:** Ricardo Daniel Teixeira Gonçalves
**Course:** Elements of Artificial Intelligence and Data Science (EIACD)

---

## 1. Introduction and Project Objective

This project, developed for Assignment 2 of the "Elements of Artificial Intelligence and Data Science" (EIACD) course at the University of Porto, focuses on building a complete Machine Learning (ML) pipeline. The primary goal is to predict whether a student will pass or fail their final exam, serving as an early intervention tool to identify at-risk students and enable proactive support.

The system utilizes an adapted real-world dataset from Cortez and Silva (2008), which includes academic, demographic, and social features of Portuguese secondary school students.

---

## 2. Dataset

The dataset, `student-data.csv`, combines 30 attributes from students in two Portuguese secondary schools ("GP" - Gabriel Pereira and "MS" - Mousinho da Silveira). Key attributes include:

*   **Demographic:** `sex`, `age`, `address` (urban/rural), `famsize`, `Pstatus` (parent's cohabitation).
*   **Parental Background:** `Medu` (mother's education), `Fedu` (father's education), `Mjob`, `Fjob`.
*   **School-related:** `reason` (for choosing school), `guardian`, `traveltime`, `studytime`, `failures` (past), `schoolsup`, `famsup`, `paid` (extra classes), `activities`, `nursery`, `higher` (wants higher education), `absences`.
*   **Social/Lifestyle:** `internet`, `romantic`, `famrel` (family relations), `freetime`, `goout`, `Dalc` (workday alcohol), `Walc` (weekend alcohol), `health`.
*   **Target Variable:** `passed` (originally G3 grade, transformed to binary yes/no).

A detailed data dictionary is available within the notebook (Section 2.1).

---

## 3. Methodology

The project follows a standard machine learning pipeline:

1.  **Data Loading & Initial Exploration:**
    *   Import necessary libraries (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Imblearn).
    *   Load the dataset.

2.  **Exploratory Data Analysis (EDA):**
    *   Data dictionary review and dataset preview (`.head()`, `.info()`, `.describe()`).
    *   Analysis of `school` variable (found imbalanced and subsequently dropped).
    *   Feature correlation analysis (heatmap) identifying relationships like `Medu`/`Fedu` and `Dalc`/`Walc`.
    *   Detailed analysis of `age`, `absences`, and `studytime` and their relationship with the `passed` outcome.

3.  **Data Cleaning & Preprocessing:**
    *   Checked for missing values and duplicates (none found).
    *   Outlier removal: Students aged 22 were removed based on EDA.
    *   **Encoding:**
        *   Binary encoding for 'yes'/'no' features.
        *   One-Hot Encoding for other nominal categorical features (`Mjob`, `Fjob`, `reason`, `guardian`, `sex`, `address`, `famsize`, `Pstatus`), using `drop_first=True`.

4.  **Feature Engineering:**
    *   `avgEdu`: Created by averaging `Medu` and `Fedu`.
    *   `student_support`: Created by summing binary `famsup` and `schoolsup`.

5.  **Feature Reduction:**
    *   Original features used for feature engineering (`Medu`, `Fedu`, `famsup`, `schoolsup`) were dropped.
    *   Features deemed not relevant or used only for EDA (`school`, `abs_cat`, `study_cat`) were dropped.
    *   PCA was performed for analysis but not used for dimensionality reduction in the final models.

6.  **Class Balance Assessment:**
    *   The target variable `passed` was found to be imbalanced (67% Pass, 33% Fail).
    *   SMOTE (Synthetic Minority Over-sampling Technique) was chosen to address this during model training.

7.  **Model Training, Tuning & Evaluation:**
    *   **Train-Test Split:** Data was split into training (70%) and testing (30%) sets, stratified by the target variable.
    *   **Pipelines:** `ImbPipeline` from `imblearn` was used, incorporating SMOTE followed by a classifier.
    *   **Classifiers Evaluated:**
        *   Decision Tree
        *   Logistic Regression (with StandardScaler)
        *   K-Nearest Neighbors (KNN) (with StandardScaler)
        *   Support Vector Classifier (SVC) (with StandardScaler)
        *   Random Forest
        *   MLPClassifier (Neural Network) (with StandardScaler)
    *   **Initial Evaluation:** Models were first evaluated with default parameters.
    *   **Hyperparameter Tuning:** `GridSearchCV` was used with 10-fold cross-validation, optimizing for F1-score. This was repeated 20 times for robust evaluation metrics for tuned models.
    *   **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and AUC (Area Under ROC Curve). Confusion matrices and ROC curves were plotted.

8.  **Feature Importance Analysis:**
    *   Analyzed for Decision Tree, Random Forest (using feature\_importances\_) and Logistic Regression (using coefficients) based on models trained on the full dataset (before tuning).

9.  **Model Demonstration:**
    *   A random student's data was used to demonstrate predictions across all trained (tuned) models.

10. **Comparative Analysis with Original Article:**
    *   The project's methodology (SETUP C: no prior grades used) and results were compared to the original Cortez & Silva (2008) study.

---

## 4. Key Findings & Results

*   **EDA Insights:**
    *   Past `failures` and `absences` were identified as potentially strong predictors.
    *   `studytime` showed a stronger correlation with success than `absences`.
    *   Parental education (`Medu`, `Fedu`) showed a moderate correlation and was combined into `avgEdu`.
*   **Model Performance (After Tuning & SMOTE):**
    *   Tuning did not consistently improve performance over default models and, in many cases, led to a slight decrease in F1-score and Recall.
    *   The best performing models (based on F1-score and recall on cross-validation) were:
        *   **Random Forest:** F1-Score ≈ 0.753, Recall ≈ 0.785
        *   **SVC:** F1-Score ≈ 0.756, Recall ≈ 0.830
    *   Logistic Regression showed a notable improvement in AUC (+13.17%) after tuning, despite a drop in F1-score.
*   **Feature Importance:**
    *   Across models (Decision Tree, Random Forest, Logistic Regression), **`failures`** was consistently the most important predictor.
    *   Other significant features included **`absences`**, social activity (`goout`), average parental education (`avgEdu`), and whether the student wants to pursue `higher` education.
*   **Comparison with Original Article (Cortez & Silva, 2008 - SETUP C):**
    *   The current analysis aligns with the "SETUP C" methodology (no prior grades).
    *   **Consistent Findings:** Both studies highlighted `failures` and `absences` as top predictors.
    *   **Performance:** Model accuracies were slightly lower than in the original paper but followed similar patterns (Random Forest and SVM performing well, around 65-70% accuracy in the current study vs. ~70% PCC in the article).

---

## 5. How to Run

1.  Ensure you have Python 3.x installed.
2.  Install Jupyter Notebook or JupyterLab.
3.  Install the required libraries:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
    ```
4.  Place the `student-data.csv` file in the same directory as the notebook.
5.  Open and run the `stu_inte_sys.ipynb` notebook.

---

## 6. Libraries Used

*   **Data Manipulation & Analysis:** NumPy, Pandas
*   **Visualization:** Matplotlib, Seaborn
*   **Data Preprocessing & Machine Learning:** Scikit-learn (StandardScaler, PCA, train\_test\_split, GridSearchCV, StratifiedKFold, various classifiers and metrics)
*   **Resampling:** Imbalanced-learn (SMOTE, ImbPipeline)
*   **Built-in:** random, time

---

## 7. References

*   Cortez, P., & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance*. University of Minho.
*   Russell, S. J., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.).
*   Hurbans, R. (2020). *Grokking Artificial Intelligence Algorithms*.
*   Gallatin, K., & Albon, C. (2023). *Machine Learning with Python Cookbook* (2nd ed.).
*   Documentation for Pandas, Scikit-learn, Matplotlib, and Seaborn.
