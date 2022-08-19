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

Joining files from the data base created a dataframe where rows corresponded to unique transactions with the necessary features for CLV modeling; customer identifier, transaction time-stamp, and the monetary value of the transaction.


## Methods
This project focuses on solving a classification problem with a predictive data science model. The problem at hand is predicting a customer's churn status (true/false). The project uses an iterative approach to building a predictive model that's both accurate and interpretable. In the process, I iterated through several model types, feature engineering methods, and hyperparameters.

## Results
The final model significantly improved Telecom Inc.'s ability to predict which customers are likely to churn. Compared to a baseline model, the final model improved F1 score by 0.55. 
- **F1 Score: 0.80**
- **Precision:** 0.90
- **Recall:** 0.72

![Precision & Recall Curve](Visuals/Final_Model_Precision_Recall.png)

Our model balances recall and precision to meet Telecom Inc.'s business need to identify high risk customers without overpredicting churn.

![Confusion Matrix](Visuals/rf_confusion_matrix.png)

The final model makes correct predictions 95% of the time. The model is very strong at correctly predicting which customers are unlikely to churn and minimizing false positives. The model struggles more with capturing all churned customers although when it does predict churn, it's highly accurate.

![Customer Service Calls](Visuals/Customer_Service_Calls_Day_Minutes.png)

The final model is interpretable which is very useful for Telecom Inc. They can use the model to identify a given customer's churn likelihood and which factors influenced their risk profile. Armed with this information, they can take targeted interventions to save the customer. For example, if the customer has many customer service calls, Telecom Inc. can pair them with their most experienced customer service reps to give them the best experience possible. If the customer has a lot of day-time usage, we can offer them a discount on the service. For customers that aren't at risk of churning, they can withhold costly interventions.


## Conclusions

This predictive algorythm solves many of Telecom Inc.'s business challenges related to customer churn
- **Identifies customers who are likely to churn**
- **Seperates high risk customers from low risk customers so they can focus support resources where it's most needed**
- **Identifies key features connected with churn so customer support can engage high risk customers with personalized account saving interventions like discounts or additional support.**

### Limitations & Next Steps

Additional data and model tuning can further improve the model.

- **Limitations** Our model misses 28% of churning customers. The final model is overly fit to training data because F1 score dropped significantly from training to test evaluation.
- **Next Steps** More examples of churn customers to balance classes will improve overfitting issues. Adding customer payment information will improve accuracy. I can explore black-box models to improve recall.
- **Predicting undesirable outcomes.** This modeling could identify animals that are more likely to have undesirable outcomes (e.g. Euthanasia) for targeted medical support or outreach.
 
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
