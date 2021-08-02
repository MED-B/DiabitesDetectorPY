#Description : this is a program that detect whether someone has diabites with ML and Python

#Import libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Create Title and Subtitle
st.write("""
#Diabites Detector
detect whetheer someone has diabites with ML and Python
""")
#Open and display an image
image = Image.open('pic.png')
st.image(image, caption="Caption : ML", use_column_width=True)

#Get the Data
df = pd.read_csv('diabetes.csv')

#Set A SubHeader
st.subheader('Data Informations : ')
#show the data as a table
st.dataframe(df)
#show statistics about the data
st.write(df.describe())
#show the data as a chart
chart = st.bar_chart(df)
#Split the data into independet 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values
#Split the data set into 75% trainig data and 25% testing data
X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#Get the feature input from the user
def get_user_input():
    Pregnancies = st.sidebar.slider('Pregrancies', 0, 17, 3)
    Glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    Blood_Pressure = st.sidebar.slider('Blood_Pressure', 0, 122, 72)
    Skin_Thickness = st.sidebar.slider('Skin_Thickness', 0, 99, 23)
    Insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider('Age', 18, 81, 29)

    #Store a dictionarry into a variable
    user_data = {
        'Pregnancies' : Pregnancies,
        'Glucose' : Glucose,
        'Blood_Pressure' : Blood_Pressure,
        'Skin_Thickness' : Skin_Thickness,
        'Insulin' : Insulin,
        'BMI' : BMI,
        'DPF' : DPF,
        'Age' : Age
                }
    #Transform the data into a DataFrame
    features = pd.DataFrame(user_data, index=[0])
    return features

#Store the user input into a variable
user_input = get_user_input()

# Set a subheader and diplaying the user inputs
st.subheader('User Input : ')
st.write(user_input)

#create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Show th e model metrics
st.subheader('Model Test Accuracy Score : ')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

#Store the model predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

#Set a subheader displaying the predictio for the user data
st.subheader('Classification')
st.write(prediction)
