import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

def generate_synthetic_data(n_samples, n_features, centers):
    """Generates synthetic data using make_blobs from sklearn."""
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=42)
    return X, y

def plot_data(X, y=None, title=""):
    """Plots the data with an optional label coloring."""
    fig, ax = plt.subplots()
    if y is not None:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k')
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
    else:
        ax.scatter(X[:, 0], X[:, 1], marker='o', edgecolor='k')
    ax.set_title(title)
    return fig

# Streamlit app setup
st.title('EM Algorithm Demonstration with GMM')

# Sidebar for user inputs
st.sidebar.header('Configuration')
n_samples = st.sidebar.slider('Number of samples', min_value=100, max_value=1000, value=300, step=50)
n_features = st.sidebar.slider('Number of features', min_value=2, max_value=3, value=2, step=1)
n_clusters = st.sidebar.slider('Number of clusters', min_value=2, max_value=5, value=3, step=1)

# Generate data based on user input
X, y = generate_synthetic_data(n_samples, n_features, n_clusters)

# Display the generated data
st.subheader('Generated Data')
st.pyplot(plot_data(X, y, "Generated Data"))

# Fit Gaussian Mixture Model using EM algorithm
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# Display the clustering result
st.subheader('GMM Clustering Result')
st.pyplot(plot_data(X, labels, "GMM Clustering"))

# Display the parameters of the GMM
st.subheader('GMM Parameters')
st.write("Means of each component:\n", gmm.means_)
st.write("Covariances of each component:\n", gmm.covariances_)
st.write("Weights of each component:\n", gmm.weights_)
