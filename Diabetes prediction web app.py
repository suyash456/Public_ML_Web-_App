
import pandas as pd
import numpy as np
import pickle
import streamlit as st

#loading the saved model 
loaded_model = pickle.load(open('C:/Deploying Machine Learning Model/trained_model.sav','rb'))

def diabetes_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction =loaded_model.predict(input_data_reshaped)

    print(prediction)

    if (prediction[0]== 0):
      return("The Person is NOn-diabetic")
    else:
      return("The Person is Diabetic")
  

def main():
    
    st.title("Diabetes Predictive System")
    
    
    Pregnancies = st.text_input('Number Of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('SkinThickness Value')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Insulin = st.text_input('Insulin Level')
    Age = st.text_input('Age of Person')
    
    #code for prediction 
    
    diagnosis= ''
    
    #creating a button for prediction
    
    if st.button('Diabeties Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,BMI,DiabetesPedigreeFunction,Insulin,Age])
        
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()
    
    