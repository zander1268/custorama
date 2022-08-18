import streamlit as streamlit
import pandas as pd
import matplotlib.pyplot as plt 

st.title("Data Summary")
#Load data
@st.cache
def load_data(path):
    pen_data = pd.read_csv(path, index_col=0)
    return pen_data

penguins = load_data('./Data/penguins.csv')

