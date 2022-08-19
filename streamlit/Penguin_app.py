import streamlit as st
from PIL import Image
#Inital imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lifetimes
from lifetimes.plotting import plot_period_transactions, plot_calibration_purchases_vs_holdout_purchases
from lifetimes import BetaGeoFitter, GammaGammaFitter
from datetime import timedelta
from datetime import datetime
from dateutil import parser
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.utils import calibration_and_holdout_data
#Functions
    #clean csv
def clean_transaction_csv(transaction_df,datetime_col,customer_id_col,monetary_value_col):
    #Remove tansactions less than or equal to zero
    sub_transaction_df = transaction_df[transaction_df[monetary_value_col]>0]
    #Subset to only repeat customers
    repeat_cust_ID = pd.DataFrame(sub_transaction_df.groupby(customer_id_col)[datetime_col].nunique())
    repeat_cust_ID = list(repeat_cust_ID[repeat_cust_ID[datetime_col]>1].index)
    sub_transaction_df = sub_transaction_df[sub_transaction_df[customer_id_col].isin(repeat_cust_ID)]
    #convert datetime_col to datetime64
#    sub_transaction_df[datetime_col] = pd.to_datetime(sub_transaction_df[datetime_col])
    #return a df with only repeat customers with orders over 0.00 monetary value
    return sub_transaction_df
    #class to return a calibration and holdout df
class df_ch():
    def __init__(self,transaction_df=None,customer_id_col=None,datetime_col=None,monetary_value_col=None):
        #initialized attributes
        self.transaction_df = transaction_df
        self.customer_id_col=customer_id_col
        self.datetime_col=datetime_col
        self.monetary_value_col=monetary_value_col
        #save off more attributes
        self.min_obs_date = parser.parse(transaction_df[datetime_col].min())
        self.max_obs_date = parser.parse(transaction_df[datetime_col].max())
        self.eval_period = np.round(((self.max_obs_date-self.min_obs_date).days * (1/3))) #one third of total range
        self.max_calib_date = self.max_obs_date - timedelta(days=self.eval_period)  
        self.calib_range_days = (self.max_calib_date - self.min_obs_date).days
    def df_ch_getdf(self):
        df = calibration_and_holdout_data(
        transactions = self.transaction_df, 
        customer_id_col=self.customer_id_col,
        datetime_col=self.datetime_col,
        monetary_value_col=self.monetary_value_col,
        calibration_period_end = self.max_calib_date, 
        observation_period_end = self.max_obs_date, 
        freq = "D")
        return df
    #function to capture RMSE for a BGF model
def bgf_rmse(ch,bgf):
    df_ch = ch.df_ch_getdf()
    df_ch["n_transactions_holdout_real"] = df_ch["frequency_holdout"]
    y_true = df_ch["n_transactions_holdout_real"]
    y_pred = bgf.predict(t=ch.eval_period, frequency=df_ch['frequency_cal'],
                         recency=df_ch['recency_cal'],
                         T=df_ch['T_cal'])

    return mean_squared_error(y_true,y_pred,squared=False)
    #Get real and predicted values from a bgf model
def bgf_real_v_pred_df(ch,bgf):
    rfm_cal_holdout = pd.DataFrame()
    ch_df = ch.df_ch_getdf()
    rfm_cal_holdout["n_transactions_cal_real"]  = ch_df["frequency_cal"] + 1 #Total calibration days with purchases = calibration frequency + 1
    rfm_cal_holdout["n_transactions_holdout_real"]  = ch_df["frequency_holdout"] #Total validation days with purchases = validation frequency
    # the predicted number of transactions
    rfm_cal_holdout["n_transactions_holdout_pred"] = bgf.predict(t=ch.eval_period, 
                                                    frequency=ch_df['frequency_cal'], 
                                                    recency=ch_df['recency_cal'], 
                                                    T=ch_df['T_cal'])
    return rfm_cal_holdout[["n_transactions_cal_real","n_transactions_holdout_real", "n_transactions_holdout_pred"]]
#function to return predicted # transactions for given customer in evaluation period
def samp_cust_pred_trans(df_ch,sample_customer_id,eval_period):
    sample_customer = df_ch.loc[sample_customer_id]
    n_transactions_pred = bgf.predict(t=eval_period,
                                  frequency=sample_customer['frequency_cal'], 
                                  recency=sample_customer['recency_cal'], 
                                  T=sample_customer['T_cal'])
    return(n_transactions_pred)
#Header
st.image("Images/windjammer_logo.jpg", use_column_width="always")
st.title("How much are your customers worth?")
st.header("Upload your transaction data")

#User inputs
    #File uploader
uploaded_file = st.file_uploader("Select your transaction csv")
@st.cache
def load_data(file):
    df = pd.read_csv(file)
    return df
if uploaded_file is not None:
    transaction_df = load_data(uploaded_file)
    st.write(transaction_df)
    column_names = list(transaction_df.columns)
        
#if uploaded_file is not None:
 #    # Can be used wherever a "file-like" object is accepted:
  #   transaction_df = pd.read_csv(uploaded_file)
   #  st.write(transaction_df)
    ##Identify how columns are titled in your csv

customer_id_coloumn = str(st.selectbox(label="Customer identifier column header",options=column_names))
datetime_coloumn = str(st.selectbox(label="Transaction date stamp column header",options=column_names))
monetary_value_coloumn = str(st.selectbox(label="Transaction value column header",options=column_names))

#saved variables from customer inputs
ch_df_obj = df_ch(transaction_df,customer_id_coloumn,datetime_coloumn,monetary_value_coloumn)
ch_df = ch_df_obj.df_ch_getdf()
full_rfm_summary = summary_data_from_transaction_data(transactions=transaction_df,
                                                      customer_id_col=customer_id_coloumn,
                                                      datetime_col=datetime_coloumn,
                                                      monetary_value_col=monetary_value_coloumn)

repeat_transaction_df = clean_transaction_csv(transaction_df,datetime_coloumn,customer_id_coloumn,monetary_value_coloumn)

repeat_rfm_summary = summary_data_from_transaction_data(transactions=repeat_transaction_df,
                                                      customer_id_col=customer_id_coloumn,
                                                      datetime_col=datetime_coloumn,
                                                      monetary_value_col=monetary_value_coloumn,)
#Train-test eval
st.header("Model Fitting")
button1 = st.button("Evaluate model fit")
if button1:    
    #Iniatialize bgf model
    bgf = BetaGeoFitter(penalizer_coef=0)
    #Fit model to ch_df
    bgf.fit(
        frequency = ch_df["frequency_cal"], 
        recency = ch_df["recency_cal"], 
        T = ch_df["T_cal"],   
        weights = None,  
        verbose = False)
    #Return rmse for bgf model
    model_rmse = bgf_rmse(ch_df_obj,bgf)
    st.write(f'Model is accurate to within {round(model_rmse,ndigits=3)} purchases over {int(ch_df_obj.eval_period)} days')
#ClV Predictions
st.header("CLV Predictions")
    #slider
prediction_period = st.slider("How many months in the future do you want to predict?",3,36,3)
    #Run model button
button2 = st.button("Return CLV predictions")
if button2: 
    gg = GammaGammaFitter(penalizer_coef = 0.001)
    bgf = BetaGeoFitter(penalizer_coef=0)
    bgf.fit(
        frequency = full_rfm_summary["frequency"], 
        recency = full_rfm_summary["recency"], 
        T = full_rfm_summary["T"],   
        weights = None,  
        verbose = False)
    gg.fit(repeat_rfm_summary['frequency'],repeat_rfm_summary['monetary_value'])
    ltv_predictions = gg.customer_lifetime_value(bgf,
    full_rfm_summary['frequency'],
    full_rfm_summary['recency'],
    full_rfm_summary['T'],
    full_rfm_summary['monetary_value'],
    time=prediction_period, # months
    discount_rate=0.0, # 
    freq ="D")
    st.write(f'Total revenue from return customers in next {prediction_period} months {np.round(sum(ltv_predictions))}')
    st.write(pd.DataFrame(ltv_predictions))
#Columns
col11, col12 = st.columns(2)
col11.subheader("Column 1")
col12.subheader("Column 2")

col21, col22, col23 = st.columns([3,2,1])
col21.write("Large column Text will wrap around if there is enough space")
col22.write("medium column")
col23.write("small column")
#Markdown
st.markdown("Markdown **syntax** *works*")
'Markdown'
'## Magic'
st.write('<h2 style="text-align:center">Text aligned with Html</h2>',)
#Widgets

check = st.checkbox("Please check this box")
if check:
    st.write("You checked box")
else:
    st.write("The box was not checked")
#Radio button
st.subheader("Radio Button")
animal_options = ["Cats","Dogs","Pigs"]
fav_animal = st.radioxf("Which animal is your favorite?",animal_options)
button2 = st.button("Submit animal")
if button2:
    st.write(f'You selected {fav_animal} as your fav animal')
#Multi select
like_animals = st.multiselect("Which animals do you like?",animal_options)
st.write(like_animals)
st.write(f'The animal you liked first was {like_animals[0]}')
#slider
num_pets = st.slider("How many pets is too many?",2,20,2)
#Text input
pet_name = st.text_input("What is your pets name?",value="I don't have a pet")
st.write(pet_name)

#Sidebar
st.sidebar.title("Sidebar")
side_button = st.sidebar.button("Press Me")
if side_button:
    st.write("Button was pressed")



