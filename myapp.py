import streamlit as st
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Predict what your mile PR will be by the end of track season (do this after xc season) - made by jon hogg
This app uses data from the 2020/2021 DGN XC/Track Team to predict what your mile PR will be in the next Track Season (You can use this as motivation during winter training) 
""")
st.write("""
It will not be that accurate for anyone on the extreme ends of the spectrum because I didn't have much data, but I hope its looks kinda cool regardless. I put a good 2 days of work into this :) (manually formatted all the data, used machine learning to create a regression model, created the web app elements, deployed to heroku etc...)
""")
st.write('---')
# load up the data
url = "predictions.csv"
names = ['gender', 'grade', 'threemile', 'newmile']
boston = pd.read_csv(url, names=names)
array = boston.values
q = array[:,0:3]
w = array[:,3]
X = np.array(q)
Y = np.array(w)

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Put your 3 Mile PR from XC Season in!')

def user_input_features():
    gender = st.sidebar.slider('Gender (0 for female/ 1 for male)', 0, 1, 0)
    grade = st.sidebar.slider('Grade', 9, 12, 9)
    minutes = st.sidebar.slider('Last XC Season Three Mile Pr (Minutes)', 0, 60, 15)
    seconds = st.sidebar.slider('Last XC Season Three Mile Pr (Seconds)', 0, 60, 0)
    deciseconds = st.sidebar.slider('Last XC Season Three Mile Pr (Deciseconds)', 0, 10, 0)

    threemile = (minutes*60)+seconds+(deciseconds/10)
    data = {'gender': gender,
            'grade': grade,
            'threemile': threemile,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Put the info in the sidebar on the left (I think there is a button in the top left corner)')
st.write(df)
st.write('---')

# Apply Model to Make Prediction
model = pickle.load(open('running_model.pkl', 'rb'))
prediction = model.predict(df)

st.header('Prediction of your Mile PR during next Track Season')
st.write(str(int(prediction/60))+":"+str(float(prediction)%60))
st.write('---')
