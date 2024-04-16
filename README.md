# credit_card_defaults_project


This project aims to predict how capable each (anonymized) individual is of paying off their credit card balances. 
The dataset represents default and non-default accounts of credit card clients in Taiwan from 2005. 
Using this historical data, I will try to build a predictive model that classifies whether an account will pay off its next month’s balance or default. 

The dataset has the following attributes:

ID,
LIMIT_BAL,
SEX,
EDUCATION,
MARRIAGE,
AGE,
PAY_1,
PAY_2,
PAY_3,
PAY_4,
PAY_5,
PAY_6,
BILL_AMT1,
BILL_AMT2,
BILL_AMT3,
BILL_AMT4,
BILL_AMT5,
BILL_AMT6,
PAY_AMT1,
PAY_AMT2,
PAY_AMT3,
PAY_AMT4,
PAY_AMT5,
PAY_AMT6,
dpnm

The last attribute, "dpnm" (did not pay next month) is the binary target vector: 0 for an account that did pay (non-default), and 1 for an account that did not pay (default).

EDA.py and Exploratory.ipynb both contain the same code -- however, I included the EDA.py code as a Jupyter Notebook to make it easier to include all the images/results that each block of code outputs. 
