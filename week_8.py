import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_synthetic_data(samples, features, classes, clusters):
    """Generates synthetic dataset for classification."""
    X, y = make_classification(n_samples=samples, n_features=features, 
                               n_informative=features, n_redundant=0,
                               n_clusters_per_class=clusters, n_classes=classes, random_state=42)
    return X, y

def visualize_data(X, y=None, title=""):
    """Visualizes the data with optional class coloring."""
    fig, ax = plt.subplots()
    if y is not None:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
    else:
        ax.scatter(X[:, 0], X[:, 1], edgecolor='k')
    ax.set_title(title)
    return fig

# Streamlit app setup
st.title('K-Nearest Neighbors (KNN) Classifier Demo')

# Sidebar for user inputs
st.sidebar.header('Configuration')
samples = st.sidebar.slider('Number of samples', min_value=100, max_value=1000, value=300, step=50)
features = st.sidebar.slider('Number of features', min_value=2, max_value=2, value=2, step=1) # Fixed to 2 for visualization
classes = st.sidebar.slider('Number of classes', min_value=2, max_value=4, value=3, step=1)
clusters_per_class = st.sidebar.slider('Clusters per class', min_value=1, max_value=3, value=1, step=1)
k = st.sidebar.slider('Number of neighbors (k)', min_value=1, max_value=15, value=5, step=1)

# Generate and split data
X, y = create_synthetic_data(samples, features, classes, clusters_per_class)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the generated data
st.subheader('Generated Training Data')
st.pyplot(visualize_data(X_train, y_train, "Training Data"))

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display the results
st.subheader('KNN Classification Results')
st.pyplot(visualize_data(X_test, y_pred, f"Test Data (Accuracy: {accuracy * 100:.2f}%)"))

# Display the accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
