import streamlit as st
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Run App
This app uses data from the 2020/2021 DGN XC/Track Team to predict what your mile PR will be in the next Track Season (You can use this as motivation during winter training)
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
st.sidebar.header('Put your info in!')

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
st.header('What you put in')
st.write(df)
st.write('---')

# Apply Model to Make Prediction
model = pickle.load(open('running_model.pkl', 'rb'))
prediction = model.predict(df)

st.header('Prediction of your Mile PR during next Track Season')
st.write(str(int(prediction/60))+":"+str(float(prediction)%60))
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
