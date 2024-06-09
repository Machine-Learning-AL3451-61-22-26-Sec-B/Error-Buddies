import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.title("Naive Bayesian Classifier")

st.sidebar.header("Upload your CSV data")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("## Data Preview")
    st.write(data.head())

    target_column = st.sidebar.selectbox("Select target column", data.columns)

    le_dict = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            le_dict[column] = le

    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("## Model Accuracy")
    st.write(f"Accuracy: {accuracy:.2f}")

    st.write("## Make Predictions")
    input_data = {}
    for column in X.columns:
        input_data[column] = st.number_input(f"Enter value for {column}", value=float(X[column].mean()))

    input_df = pd.DataFrame([input_data])

    for column, le in le_dict.items():
        if column in input_df.columns:
            input_df[column] = le.transform(input_df[column])

    prediction = model.predict(input_df)
    st.write("## Prediction")
    st.write(f"Predicted class: {prediction[0]}")
else:
    st.write("Please upload a CSV file to get started.")
