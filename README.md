#  CREDIT RISK PREDICTION  
### Machine Learning Project — Bankruptcy & Loan Default Risk Assessment  
**By Eve Giamminoni**

---

## Project Overview  
Credit Risk Prediction is a Machine Learning project designed to evaluate the financial risk associated with American companies and predict whether they are likely to default on a loan.  
The goal is to build a **decision-support tool for financial institutions**, helping them determine whether to approve or reject loan applications based on company financial metrics and simulated loan characteristics.

This project includes:  
- A complete ML pipeline  
- Feature engineering tailored to credit risk  
- An optimized CatBoost model  
- A fully interactive **Streamlit application** for real-time prediction  

---

##  Dataset  
**Source:** Kaggle — *American Company Bankruptcy Prediction*  
The dataset contains:  
- ~78,000 U.S. companies  
- 18 key financial ratios  
- A target variable indicating bankruptcy (1) or survival (0)

These ratios include:  
- Profitability metrics (EBITDA, Net Income, Gross Profit…)  
- Liquidity ratios (Current Liabilities, Receivables…)  
- Leverage and balance sheet structure  
- Operational performance indicators  

This dataset represents a realistic snapshot of the variables banks often examine when assessing corporate creditworthiness.

---

##  Engineered Features  
To better emulate real banking practices, two additional features were created:

### **1️.loan_amount**  
A simulated value representing how much the company requests to borrow.

### **2️. loan_ratio = loan_amount / Total_Assets**  
This ratio expresses how large the loan request is relative to the company’s assets.  
Banks commonly use loan-to-asset or loan-to-value ratios to quantify credit exposure risk.

These engineered variables significantly improve model realism and decision accuracy by linking financial health to borrowing behavior.

---

##  Machine Learning Workflow

A full comparative modeling strategy was implemented to identify the most reliable classifier for bankruptcy prediction.  
The modeling phase included baseline evaluation, imbalance handling, advanced modeling, and hyperparameter optimization.

---

###  1. Baseline Models
To establish reference performance levels, three simple classifiers were trained:

- **Dummy Classifier** — represents the naïve baseline  
- **Decision Tree Classifier**  
- **Random Forest Classifier**

These baselines provided an initial understanding of model difficulty and dataset imbalance.

---

###  2. Handling Class Imbalance
Multiple strategies were tested to improve minority class recall:

- **class_weight = "balanced"** (sklearn)
- **SMOTE (oversampling)**
- **imbalanced-learn techniques** applied to Random Forest  
  (resampling + balanced ensembles)

These approaches improved sensitivity to bankruptcy cases, but performance remained limited.

---

###  3. Advanced Algorithms
To explore more powerful models, two gradient-boosting approaches were tested:

- **CatBoost Classifier**  

CatBoost delivered significantly better performance on tabular financial data, even before tuning, and handled imbalance effectively.

---

###  4. Hyperparameter Optimization (Optuna)
Optuna was used to search the CatBoost parameter space efficiently:
- **CatBoost with Optuna hyperparameter optimization**

The Optuna-tuned CatBoost provided the best F1 performance on the minority class.

---

###  5. Threshold Adjustment
Because default probabilities are asymmetric and banks tend to be risk-averse, the default decision threshold was tuned.

A value of:

### **➡️ Threshold = 0.40**

produced the best balance between:
- detecting bankrupt companies  
- minimizing false rejections  
- aligning with typical banking risk tolerance  

---

### Final Selected Model
**CatBoost + Optuna Optimization + Threshold = 0.40**

This configuration achieved:

- **F1-score (bankrupt): ~0.55**  
- **F1-score (healthy): ~0.97**  
- **Accuracy: ~0.94** 

And delivered stable, interpretable, and practical predictions suitable for loan default assessment.

---

##  Default Probability and Loan Recommendation  
For any company input, the system computes:

- Probability it will default  
- Probability it will remain solvent  
- A final recommendation:  
  - **Loan Approved** (low risk)  
  - **Loan NOT Recommended** (high risk)

This transforms Creditrix from a model into a **practical credit-scoring system**, usable by analysts and financial institutions.

---

## Streamlit Application  
A full interactive interface was developed to make the model usable in real time.

###  App Capabilities  
Users can:  
- Enter all key financial ratios  
- Input a custom loan amount  
- Automatically compute the loan_ratio  
- View predicted repayment & default probabilities  
- Receive a loan recommendation based on threshold logic  

The interface includes a modern UI design, with responsive sections and collapsible explanations for readability.

---

##  Feature Importance  
The Streamlit app displays a horizontal bar chart visualizing CatBoost feature importance.  
Key influential features include:  
- loan_ratio  
- loan_amount  
- Operating Expenses  
- Market Value  
- Total Liabilities  

This helps interpret the model and understand what drives predictions.

---

### Programming Environment
- **Python 3.9.6**
---
  
### Technical Stack
**pandas**
**numpy**
**matplotlib.pyplot**
**seaborn**

**train_test_split**
**DecisionTreeClassifier**
**RandomForestClassifier**
**DummyClassifier**
**LogisticRegression**
**LabelEncoder**

**classification_report**
**confusion_matrix**
**accuracy_score**
**precision_score**
**recall_score**
**f1_score**
**roc_curve**
**roc_auc_score**
**auc**
**precision_recall_curve**
**average_precision_score**

**SMOTE**
**BalancedRandomForestClassifier**

**CatBoostClassifier**
**Pool**

**optuna**
**pickle**
**streamlit**
