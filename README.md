# 🏥 Medical Insurance Cost Prediction

A Machine Learning regression project that predicts **medical insurance
charges** using **Linear Regression**.

## 📊 Dataset

The dataset contains **1,338 records** and **7 original columns**:

-   `age`
-   `sex`
-   `bmi`
-   `children`
-   `smoker`
-   `region`
-   `charges` --- target variable

## 🔎 Exploratory Data Analysis

The project includes:

-   Dataset inspection with `head()`, `tail()`, `info()`, and
    `describe()`
-   Categorical analysis
-   Medical charges distribution
-   Smoking status vs. charges
-   BMI vs. charges
-   Age vs. charges
-   Children vs. charges
-   Missing-value check
-   Duplicate check

### Key Findings

-   `charges` has a right-skewed distribution.
-   `smoker` shows a strong relationship with medical charges.
-   `age` shows a positive relationship with charges.
-   `bmi` has a more scattered relationship with charges.
-   `children` does not show a strong direct relationship with charges
    by itself.
-   No missing values were found.
-   No duplicate rows were found.

## 🧹 Preprocessing

Categorical features were converted using One-Hot Encoding:

``` python
df = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True
)
```

`drop_first=True` was used to avoid the Dummy Variable Trap.

No outlier removal was performed because the high charge values were
treated as potentially valid observations.

## 🤖 Model

**Linear Regression**

-   Train/Test split: **80/20**
-   `random_state=42`
-   Target: `charges`

## 📈 Evaluation

  Metric            Result
  -------- ---------------
  MAE             4,177.05
  MSE        35,478,020.68
  RMSE            5,956.34
  R²                0.8069

The model achieved an **R² score of 0.8069 (80.69%)** on the test set.

> R² is not classification accuracy. It represents the proportion of
> variance in the target explained by the regression model.

## 📌 Coefficient Analysis

The model coefficients were inspected to understand the learned
relationships.

The largest positive coefficient was:

**`smoker_yes` → \~23,077.76**

This means that, with the other model features held constant, the model
associates `smoker_yes` with a substantially higher predicted charge
relative to the reference category.

## 🛠️ Technologies

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Jupyter Notebook

## 📂 Project Structure

``` text
medical-insurance-cost-prediction/
│
├── medical_insurance_cost_prediction.ipynb
├── insurance.csv
├── README.md
└── assets/
    ├── medical-insurance-cover.png
    └── screenshots/
```

## 🚀 Workflow

``` text
Data Collection
      ↓
Exploratory Data Analysis
      ↓
Data Quality Checks
      ↓
Categorical Encoding
      ↓
Train / Test Split
      ↓
Linear Regression
      ↓
Prediction
      ↓
Model Evaluation
      ↓
Coefficient Analysis
```

## 👨‍💻 Author

**Amr Ayman**

Computer Science Student \| Data Science & AI

GitHub: https://github.com/amrayman-163
