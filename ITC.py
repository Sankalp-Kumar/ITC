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
    df = pd.read_excel(uploaded_file.name,index_col='Yr-Wk')
    df.head()
    st.subheader('Raw Data')
    st.write(df)
    train=df[['Seasonality Index','Discount Avg','Quantity']]
    print(train.shape)
    
    def createXY(dataset,n_past,n_future):
        dataX = []
        dataY = []
        for i in range(n_past, len(dataset)-n_future+1):
            #print(dataset.iloc[i - n_past:i, 0:dataset.shape[1]])
            dataX.append(dataset.iloc[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset.iloc[i+n_future-1:i+n_future,-1])
        return np.array(dataX),np.array(dataY)
    xtrain,y=createXY(train,1,0)
    st.write(xtrain.shape,y.shape)
    
    
