# Credit Default Risk Analysis

**Author:** Brian Owino Abayo  
**Status:** Portfolio project  
**Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

## Project overview

This project explores the factors associated with credit-card payment default and builds a baseline model to estimate default risk. It is designed as a practical junior data-science portfolio project: the workflow is reproducible, the evaluation focuses on more than accuracy, and the dashboard makes the results easier to inspect.

The dataset is the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository. It contains 30,000 Taiwan credit-card client records and a binary next-month default outcome. It is useful for demonstrating a credit-risk workflow, but it is **not Kenyan borrower data and should not be treated as representative of Pezesha customers**.

## Questions explored

1. How common is next-month default in this dataset?
2. How do repayment status, bill amounts and payment amounts vary between default and non-default groups?
3. How well does a baseline logistic-regression model distinguish the two classes?
4. What happens to precision and recall as the decision threshold changes?
5. What are the limitations of using this dataset and model for real lending decisions?

## Repository structure

```text
pezesha-credit-risk-portfolio/
├── data/
│   └── README.md
├── notebooks/
│   └── 01_credit_default_risk_analysis.ipynb
├── src/
│   └── analysis.py
├── dashboard/
│   └── app.py
├── reports/
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python src/analysis.py
streamlit run dashboard/app.py
```

The first run downloads the dataset from the UCI repository and caches it locally. An internet connection is required for the initial download.

## Evaluation approach

The model uses a stratified train/test split and a preprocessing pipeline. The numeric features are imputed and scaled; categorical repayment-status features are one-hot encoded. Logistic regression is used as an interpretable baseline.

The report includes:
- Confusion matrix
- Precision, recall and F1-score
- ROC-AUC and average precision
- Classification results at the default 0.50 threshold
- A threshold analysis to make the precision/recall trade-off visible

Accuracy is reported for completeness, but it is not treated as the only measure because the default class is less common than the non-default class.

## Important limitations

- The source data concerns credit-card clients in Taiwan, not microfinance borrowers in Kenya.
- Historical relationships do not establish causation.
- The dataset may contain social, economic and institutional biases.
- A model score is not a lending decision. Real deployment would require local validation, governance, explainability, fairness testing, monitoring, consent and human oversight.
- The project is educational and must not be used to approve or reject real applicants.

## Data source

Yeh, I. C. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository.  
https://doi.org/10.24432/C55S3H

Dataset landing page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

## Author

Brian Owino Abayo  
GitHub: https://github.com/brainowino
