import streamlit as st
import pickle

model = pickle.load(open('Prediksi_diabetes.sav','rb'))

st.title('Prediction Diabetes')

st.write('please enter the following detail for prediction')


Age = st.number_input("Masukan Umur", 0, 80, 40)

Glucose = st.number_input("Masukan data Glucose", 0, 200, 0)

Insulin = st.number_input("Masukan data Insulin", 0, 100, 10)

Pregnancies = st.number_input("Masukan data Pregnancies", 0, 10, 0)

BloodPressure = st.number_input("Masukan data BloodPressure", 0, 120, 0)

BMI = st.number_input("Masukan data BMI", 0, 50, 0)

SkinThickness = st.number_input("Masukan data SkinThickness", 0, 50, 10)

DiabetesPedigreeFunction = st.number_input("Masukan data DiabetesPedigreeFunction", 0.0, 6.2, 3.2)

Outcome = st.selectbox("Masukan data Outcome", options = ('YES', 'NO'))

if (Outcome == 'YES'):
    Outcome = 1
else:
    Outcome = 0 

#create a button to start for prediction
if st.button ('test_prediksi_diabetes') :
    diabetes_predict = model.predict([Age, Glucose, Insulin, Pregnancies, BloodPressure, BMI, SkinThickness, DiabetesPedigreeFunction, Outcome])
    if (diabetes_predict[0] == 0):
      st.success = 'Patients Affected by Diabetes '
    else :
      st.warning = 'Patient is not Affected by Diabetes'