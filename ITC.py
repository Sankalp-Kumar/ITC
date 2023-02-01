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
    model1=tf.keras.models.load_model('LSTM_MAPE(11).h5')
    #st.write(model1.summary())
    mape=model1.evaluate(xtrain,train['Quantity'])
    st.write('MAPE Score :',mape)
    pred=model1.predict(xtrain)
#     plt.figure(figsize=(15, 7.5))
#     plt.plot(pred, color='r', label='model')

#     #plt.axvspan(train.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
#     plt.plot(train['Quantity'], label='actual')
#     plt.legend()
#     plt.show()
    train['pred'] = pred
    figure2,x2 = plt.subplots()
    plot2 = train[['Quantity', 'pred']]
        #st.write(plot2)
    plot2 = x2.plot(plot2)
    st.subheader('Showing the graph for Actual and Predicted values')
    st.pyplot(figure2)
else : 
    st.write('Please upload a file')
