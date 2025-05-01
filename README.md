# Telecom Customer Churn Prediction

## Overview
This project focuses on predicting customer churn in the telecommunications industry. Customer churn occurs when customers discontinue their service with a company. In the highly competitive telecom sector, where annual churn rates range from 15-25%, predicting and preventing customer churn is crucial for business success.

## Business Impact
- Customer retention is more cost-effective than acquiring new customers
- Early detection of potential churn helps focus retention efforts on high-risk customers
- Reducing churn directly impacts company profitability and market position

## Dataset
The project uses the [Telco Customer Churn](https://www.kaggle.com/bhartiprasad17/customer-churn-prediction/data) dataset, which includes:
- Customer churn status
- Service subscriptions (phone, internet, security, etc.)
- Account information (tenure, contract, payment method)
- Demographic data (gender, age, partner status, dependents)

## Key Findings

### Customer Churn Analysis
- 26.6% of customers switched to another provider
- Gender has negligible impact on churn rates
- Contract type significantly affects churn:
  - 75% of month-to-month customers churned
  - 13% of one-year contract customers churned
  - 3% of two-year contract customers churned

### Payment Methods
- Electronic check users have higher churn rates
- Automatic payment methods (credit card, bank transfer) show lower churn rates

### Service Impact
- Fiber optic customers show higher churn rates compared to DSL
- Customers without tech support are more likely to churn
- Lack of online security is a major factor in customer churn

### Customer Demographics
- Senior citizens show higher churn rates
- Customers without dependents are more likely to churn
- Paperless billing customers have higher churn rates

### Financial Factors
- Higher monthly charges correlate with increased churn
- New customers (shorter tenure) show higher churn rates

## Machine Learning Implementation

### Models Evaluated
1. Logistic Regression
2. K-Nearest Neighbors
3. Naive Bayes
4. Decision Trees
5. Random Forest
6. AdaBoost
7. Gradient Boosting
8. Voting Classifier

### Final Model
A Voting Classifier combining:
- Gradient Boosting
- Logistic Regression
- AdaBoost

### Model Performance
- Final Accuracy: 84.68%
- Confusion Matrix Analysis:
  - True Negatives: 1400
  - False Positives: 149
  - False Negatives: 280
  - True Positives: 281

## Technical Implementation
- Libraries: scikit-learn, Matplotlib, pandas, seaborn, NumPy
- Model Evaluation: K-fold cross-validation
- Feature Engineering: Comprehensive preprocessing of categorical and numerical features

## Future Improvements
- Hyperparameter tuning
- Advanced feature engineering
- Ensemble method optimization

## Contact
For feedback or questions, please reach out at pradnyapatil671@gmail.com

## Author
Hi, I'm Shivanshu!
- AI Enthusiast
- Data Science & ML practitioner 