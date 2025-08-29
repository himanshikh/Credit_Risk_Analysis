# Credit Risk Analysis — Predicting Loan Default

## Overview
This project applies **Machine Learning models** to predict whether a borrower will **default on a loan within 2 years**, using the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset.  

The dataset is highly **imbalanced** — only ~7% of borrowers default. This makes traditional accuracy misleading, so we focus on **balanced accuracy, recall, and imbalanced classification metrics**.

---

## Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE, Ensemble models)  
- Matplotlib / Seaborn  

---

## Dataset
- **Source:** Kaggle – Give Me Some Credit  
- **Size:** ~150,000 records  
- **Features:** Credit utilization, age, income, debt ratio, late payment history, dependents  
- **Target:**  
  - `1` = High-Risk (default within 2 years)  
  - `0` = Low-Risk (no default)  

---

## Approach
1. **Data Preprocessing**
   - Removed missing values  
   - Train-test split (80:20)  
   - Feature scaling with StandardScaler  

2. **Baseline Model**
   - Logistic Regression → poor recall on high-risk class  

3. **Resampling**
   - SMOTE (oversampling) → slight improvement in recall  

4. **Ensemble Models**
   - Balanced Random Forest  
   - Easy Ensemble Classifier  

---

## Results

### Balanced Random Forest
- **Balanced Accuracy:** ~0.75  
- **High-Risk Recall:** 0.65  
- **High-Risk Precision:** 0.26  
- **Confusion Matrix:**



---

## Insights
- **Logistic Regression** failed to handle imbalance (ignored high-risk class).  
- **Balanced Random Forest** improved precision but missed more defaulters.  
- **Easy Ensemble Classifier** achieved the **best recall (0.73)**, detecting more high-risk borrowers.  

In financial applications, **recall is more important**: missing a risky borrower (false negative) is worse than flagging a safe one (false positive).  
Therefore, the **Easy Ensemble Classifier is the preferred model** for credit risk detection.  

---

## Conclusion
- Built a complete ML pipeline for credit risk prediction.  
- Demonstrated the impact of **resampling + ensemble methods** on imbalanced datasets.  
- Final model (Easy Ensemble) achieved:  
- **Balanced Accuracy ~0.76**  
- **High-Risk Recall ~0.73**  


