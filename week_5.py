import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Streamlit application setup
st.title('Naive Bayes Classifier Demo')

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

st.header('Classify New Iris Data')

# Input form for new data
with st.form(key='input_form'):
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.0)
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2)
    
    submit_button = st.form_submit_button(label='Classify')

# Classification of new input data
if submit_button:
    new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = nb_classifier.predict(new_data)
    prediction_proba = nb_classifier.predict_proba(new_data)
    
    st.write(f"Predicted Class: {iris.target_names[prediction[0]]}")
    st.write("Prediction Probabilities for each class:")
    for idx, class_name in enumerate(iris.target_names):
        st.write(f"{class_name}: {prediction_proba[0][idx]:.4f}")

