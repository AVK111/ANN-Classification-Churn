import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Loading trained model
model=tf.keras.models.load_model('model.h5')

## Load encoders and scalers
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

## Streamlit app
st.title('Customer Churn Prediction')
st.write('Enter customer details to predict churn probability.')
## Input fields
credit_score=st.number_input('Credit Score',min_value=300,max_value=850,value=600)
geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance',min_value=0.0,value=1000.0)
estimated_salary=st.number_input('Estimated Salary',min_value=0.0,value=50000.0)
num_of_products=st.slider('Number of Products',1,4)
tenure=st.slider('Tenure (years)',0,10)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

input_data=({
    'CreditScore':credit_score,
    'Gender':label_encoder_gender.transform([gender])[0],
    'Age':age,
    'Balance':balance,
    'Tenure':tenure,
    'NumOfProducts':num_of_products,
    'HasCrCard':has_cr_card,
    'IsActiveMember':is_active_member,
    'EstimatedSalary':estimated_salary
})

input_data = pd.DataFrame([input_data])


## one hot encoding geography
geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Combine all inputs
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

## Reordering columns to match training data
input_data = input_data[scaler.feature_names_in_]

st.write(input_data.dtypes)
st.write(input_data)

##Scale the input data
input_data_scaled=scaler.transform(input_data)

## Prediction churn
prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

if prediction_proba>0.5:
    st.write(f'The customer is likely to churn with a probability of {prediction_proba:.2f}')
else:
    st.write(f'The customer is unlikely to churn with a probability of {prediction_proba:.2f}')
