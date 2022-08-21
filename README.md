# Windjammer Consulting: Customer Lifetime Value Engine

**Author**: [Alex FitzGerald](https://www.linkedin.com/in/alex-fitzgerald-0734076a/)

![windjammer header](visuals/windjammer_logo.jpg)

## Overview
This project creates a user-friendly customer lifetime value (CLV) prediction engine able to take in transaction data and return important CLV predictions
for the merchant's customer base over a selected period of time in the future.

## Business Problem
Olist, the largest e-commerce department store in Brazil, allows small business merchants to sell their goods through a single marketplace. Olist makes their money taking a commision on orders placed on their marketplace. They want to help their merchants sell more products and retain those merchants on their marketplace by providing top-notch technical solutions. 
Olist has decided to create a user friendly app to help their merchants better understand the customers who buy their products through the Olist marketplace. Empowering their merchants with customer insights will help merchants improve their strategy, increasing sales and commisions through Olist. The addition of a customer insight tool will also improve the retention of merchants on the Olist network. 

To accomplish this project, Olist has hired Windjammer Consulting, a data-science consulting firm.

To test out the app for accuracy, Olist has provided a full transaction data set of purchases from all vendors and asked Windjammer Consulting to return CLV predictions with minimal error (RMSE).
If the model passes muster, they want Windjammer Consulting to create an application that allows merchants to upload their own data and retrieve predictions.

See more on website: www.olist.com

## Data

The data for this project was sourced from a [Olist data base provided to Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) containing information on 99,441 online transactions from 96,096 unique customers. Transactions included in the data ranged 773 days, from 2016-09-04 to 2018-10-17.

Joining files from the data base created a dataframe where rows corresponded to unique transactions with the necessary features for CLV modeling; a unique customer identifier, transaction time-stamp, and the monetary value of the transaction.


## Methods
CLV is the present value of all future cash flows of a current customer. Given the application of the model, calculating the value of a customer n periods into the future, we'll estimate a customer's probability of being alive n periods of time in the future and use this parameter to discount the frequency and monetary value of their purchases. Thus the forumla becomes CLV at time T = (Transactions per Period X Average Value Per Transaction X Probability of Being Active at Time T).

I used a heirarchical modeling approach estimate each of these three elements of CLV. 

**Beta geometric negative binomial distribution (BG/NBD) model**
- Used to estimate **Probability of Being Active at Time T** and **Transacitons per Period**
- After each transaction, an individual has a p_i probability of de-activating (never buying again)
- Each individual, i, has a hidden transaction per period rate (lambda_i) and probability of de-activating following a purchase (p_i)
- Individual lambda_i and p_i parameters are contrained by population wide Gamma and a Beta distribution respectively
- Individuals purchases follow a Poisson process with rate lambda_i*t 

**Gamma gamma model (GG)**
- Used to estimate **Average Value Per Transaction**
- For any given customer, total spend across x transactions is distributed gamma



## Results
**BG/NBD Model**
- 0.142 RMSE (transactions)
**GG Model**
- 6.62 RMSE (average transaction value)

![Dummy model performance](visuals/dummy_model_performance_repeat_purchasers.png)

![Final BG/NBD model performance](visuals/final_bgf_model_performance.png)


## Conclusions
TBD

### Limitations & Next Steps

TBD

 
## For More Information

See the full analysis in the [Jupyter Notebook](./Code/modeling.ipynb) or review this [presentation](./Churn_Buster_presentation.pdf).

For additional info, contact Alex FitzGerald

## Repository Structure

```
├── Code
│   ├── EDA_notebook.ipynb
│   ├── modeling.ipynb
├── Data
│   ├── churn-in-telecoms-dataset.csv
│   ├── cleaned_data.csv
│   ├── feature_importances.csv
├── Visuals
├── Churn_Buster_presentation.pdf
└── README.md
```
