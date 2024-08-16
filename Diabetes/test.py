import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

model = pickle.load(open("/home/suyog/Desktop/Projects/model.pkl","rb"))
scaler = pickle.load(open("/home/suyog/Desktop/Projects/scalerfile2.sav","rb"))


# Creating a Streamlit app
def main():
  
  st.title('Diabetes Prediction')

  Pregnancies = st.text_input('Number of times pregnant')
  Glucose = st.text_input('Plasma glucose concentration a 2 hours in an oral glucose tolerance test')
  BloodPressure = st.text_input('Diastolic blood pressure (mm Hg)')
  SkinThickness = st.text_input('Triceps skin fold thickness (mm)')
  Insulin = st.text_input('2-Hour serum insulin (mu U/ml)')
  BMI = st.text_input('Body mass index (weight in kg/(height in m)²)')
  DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function')
  Age = st.text_input('Age of the Person')

  if st.button('Predict'):
    input_data = (Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    for i in input_data:
       i = float(i)
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    std_data = scaler.transform(input_data_reshaped)
    print(std_data)
    result = model.predict(std_data)
    print(result)
    if (result[0] == 0):
        st.write('The person is not diabetic')
    else:
        st.write('The person is diabetic')


if __name__ == '__main__':
  main()