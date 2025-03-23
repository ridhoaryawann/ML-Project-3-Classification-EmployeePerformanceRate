import streamlit as st 
import joblib
import numpy as np 

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Employee Performance Predictions")
st.write("Predicts employee performance score based on their attributes")
st.divider()

years = st.number_input("Enter the years at company", min_value=0, max_value=15, value=2)
salary = st.number_input("Enter monthly salary",min_value=1000, max_value=100000, value=10000)
overtime = st.number_input("Enter overtime hours",min_value=0,max_value=100,value=0)
promotion = st.number_input("Enter how many times worker is being promoted",min_value=0,max_value=10,value=0)
satisfaction = st.number_input("Enter employee satisfaction score",min_value=0.0, max_value=5.0, value=3.0)

x = [years, salary, overtime, promotion, satisfaction]

st.divider()

predictbutton = st.button("Predict performance score")

if predictbutton:
    x1 = np.array(x)

    x_array = scaler.transform([x1])

    prediction = model.predict(x_array)[0]

    st.balloons()
    
    st.write(f"Prediction for the employee performance score is {prediction}")
else:
    st.write("Please use the button to estimate employee performances")