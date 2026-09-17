# Dataset Description

## Source
PaisaBazaar Banking Dataset (credit scoring data), publicly available on Kaggle.

## Overview
- **Rows:** 100,000 customer-month observations
- **Columns:** 28 original features
- **Unique Customers:** 12,500
- **Time Span:** 8 months per customer
- **Target:** No explicit fraud label. Synthetic `Is_Fraud` flag engineered using domain rules.

## Original Features

| Feature | Type | Description |
|---------|------|-------------|
| ID | String | Unique record identifier |
| Customer_ID | String | Unique customer identifier |
| Month | Int | Month of observation (1-8) |
| Name | String | Customer name |
| Age | Int | Customer age |
| SSN | Float | Social security number (masked) |
| Occupation | Categorical | 14 occupations (Engineer, Doctor, etc.) |
| Annual_Income | Float | Annual income |
| Monthly_Inhand_Salary | Float | Monthly take-home salary |
| Num_Bank_Accounts | Int | Number of bank accounts |
| Num_Credit_Card | Int | Number of credit cards |
| Interest_Rate | Int | Interest rate on credit card |
| Num_of_Loan | Int | Number of loans |
| Type_of_Loan | String | Loan types (comma-separated) |
| Delay_from_due_date | Int | Average days delayed from payment due date |
| Num_of_Delayed_Payment | Int | Number of delayed payments |
| Changed_Credit_Limit | Float | Credit limit change amount |
| Num_Credit_Inquiries | Int | Number of credit inquiries |
| Credit_Mix | Categorical | Credit mix quality (Bad/Standard/Good) |
| Outstanding_Debt | Float | Outstanding debt amount |
| Credit_Utilization_Ratio | Float | Credit utilization ratio (%) |
| Credit_History_Age | Int | Credit history age in months |
| Payment_of_Min_Amount | Categorical | Whether minimum payment is made (Yes/No) |
| Total_EMI_per_month | Float | Total EMI per month |
| Amount_invested_monthly | Float | Monthly investment amount |
| Payment_Behaviour | Categorical | Payment behavior pattern |
| Monthly_Balance | Float | Monthly balance |
| Credit_Score | Categorical | Credit score (Good/Standard/Poor) |

## Target Engineering

Fraud label created using composite rules on the **Poor credit segment**:
- **Median delayed payments** in Poor segment: 17
- **P75 credit utilization** in Poor segment: 36.21%
- **P75 credit inquiries** in Poor segment: 11

A record is flagged as `Is_Fraud = 1` if Credit_Score is Poor AND the customer exhibits at least one extreme behavioral pattern (very high delayed payments, very high utilization, or very high inquiries).

## Label Distribution
- Non-Fraud: 79,660 (79.7%)
- Fraud: 20,340 (20.3%)

## Missing Values
- Only `Payment_of_Min_Amount` had 12,007 missing values (sentinel "NM"), imputed with mode.
