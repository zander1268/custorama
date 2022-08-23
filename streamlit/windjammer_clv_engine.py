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
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import xlsxwriter

#Functions
    #clean csv
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data
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
st.image("./custorama/streamlit/Images/windjammer_logo.jpg", use_column_width="always")
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
    #column_names = ["select column...", *column_names]
    customer_id_coloumn = str(st.selectbox(label="Customer identifier column header",options=column_names))
    datetime_coloumn = str(st.selectbox(label="Transaction date stamp column header",options=column_names))
    monetary_value_coloumn = str(st.selectbox(label="Transaction value column header",options=column_names))
else:
    st.write("WAIT: Please upload transaction data that includes; transaction timestamp, transaciton value, and unique customer ID associated with order")
#saved variables from customer inputs
if uploaded_file is not None:
    check = st.checkbox("Columns selected, ready to move to modeling")
    if check:
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
        st.write("Now you're ready to select penalizer strength. Tip, start with 0.00 and work your way up till you find the lowest error.")
    else:
        st.write("WAIT: Please select your column headers")

#Train-test eval
st.header("Model Fitting")
penalizer = st.slider("Adjust penalizer strength for optimal fit",0.0,0.1,0.0,.02)
button1 = st.button("Evaluate model fit")
if button1:    
    #Iniatialize bgf model
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
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
    #Plot train and eval
        #TBD
#ClV Predictions
st.header("CLV Predictions")
prediction_period = st.slider("How many months in the future do you want to predict?",3,36,12,3)
    #Run model button
button2 = st.button("Return CLV predictions")
if button2: 
    gg = GammaGammaFitter(penalizer_coef = .001)
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
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
    #Predicted clv df
    ltv_predictions = pd.DataFrame(ltv_predictions)
    st.write(f'Total revenue from return customers in next {prediction_period} months: {np.round(ltv_predictions["clv"].sum())}')
    clv_sum_per_day = []
    date_range = [*range(1, (prediction_period)+1, 1)]
    for months in date_range:
        ltv_predictions = gg.customer_lifetime_value(
        bgf, #our best bgf model
        full_rfm_summary['frequency'],
        full_rfm_summary['recency'],
        full_rfm_summary['T'],
        full_rfm_summary['monetary_value'],
        time=months, # months
        freq ="D")
        clv_sum_per_day.append(sum(ltv_predictions))
    #Plot a line graph of cumulative CLV over the prediction period
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(date_range,clv_sum_per_day)
    ax1.set_title(f'Cumulative CLV Next {prediction_period} Months')
    ax1.set_ylabel("Cumulative CLV")
    ax1.set_xlabel("Months in The Future")
    st.pyplot(fig=fig1)
st.header("Customer Analysis")
button3 = st.button("Get CLV predictions by customer")
if button3:
    #CLV df
    gg = GammaGammaFitter(penalizer_coef = .001)
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(
        frequency = full_rfm_summary["frequency"], 
        recency = full_rfm_summary["recency"], 
        T = full_rfm_summary["T"],   
        weights = None,  
        verbose = False)
    gg.fit(repeat_rfm_summary['frequency'],repeat_rfm_summary['monetary_value'])
    ltv_predictions = gg.customer_lifetime_value(
    bgf, #our best bgf model
    full_rfm_summary['frequency'],
    full_rfm_summary['recency'],
    full_rfm_summary['T'],
    full_rfm_summary['monetary_value'],
    time=prediction_period, # months
    discount_rate=0.0, # 
    freq ="D")
    ltv_predictions_df = pd.DataFrame(ltv_predictions)
    #Predicted purchases
    prediction_period_days = prediction_period * 30
    n_predicted_purchases_base = bgf.conditional_expected_number_of_purchases_up_to_time(prediction_period_days,
                                                        full_rfm_summary['frequency'],
                                                        full_rfm_summary['recency'],
                                                        full_rfm_summary['T'])
    #Probablity alive at end of observation period
    prob_alive_now = bgf.conditional_probability_alive(full_rfm_summary['frequency'],full_rfm_summary['recency'],full_rfm_summary['T'])
    #Plot a histogram of our ltv predictions
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.hist(ltv_predictions)
    ax2.set_title(f'CLV Next {prediction_period} Months')
    ax2.set_ylabel("Customer Count")
    ax2.set_xlabel("Predicted CLV")
    st.pyplot(fig=fig2);
    joined_df = ltv_predictions_df
    joined_df["predicted_purchases"] = n_predicted_purchases_base
    joined_df["probability_alive_now"] = prob_alive_now
    joined_df.reset_index(inplace=True)
    joined_df
    df_xlsx = to_excel(joined_df)
    st.download_button(label='📥 Download Current Result',
                                data=df_xlsx ,
                                file_name= 'windjammer_clv_model.xlsx')
