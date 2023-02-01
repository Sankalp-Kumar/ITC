import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
import tensorflow as tf
import streamlit as st


st.title('DEMAND FORECASTING ON ITC DATA')

uploaded_file = st.file_uploader("Upload The Excel file here")

if uploaded_file:
    st.write("Filename: ", uploaded_file.name)
