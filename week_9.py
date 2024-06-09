import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Locally Weighted Regression algorithm
def locally_weighted_regression(x, y, query_point, tau=0.1):
    """Performs Locally Weighted Regression."""
    m = x.shape[0]
    n = x.shape[1]
    w = np.zeros((m, m))
    np.fill_diagonal(w, np.exp(-np.sum((x - query_point) ** 2, axis=1) / (2 * tau * tau)))
    theta = np.linalg.inv(x.T @ (w @ x)) @ (x.T @ (w @ y))
    return query_point @ theta

# Generate synthetic data
def generate_data(samples):
    """Generates synthetic data points."""
    x = np.linspace(0, 10, samples)
    y = np.sin(x) + np.random.normal(0, 0.1, samples)
    return x, y

# Streamlit app setup
st.title('Locally Weighted Regression (LWR) Algorithm Demo')

# Sidebar for user inputs
st.sidebar.header('Configuration')
samples = st.sidebar.slider('Number of samples', min_value=50, max_value=500, value=100, step=50)
query_point = st.sidebar.slider('Query point', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
tau = st.sidebar.slider('Bandwidth (tau)', min_value=0.01, max_value=1.0, value=0.1, step=0.01)

# Generate synthetic data
x, y = generate_data(samples)

# Perform Locally Weighted Regression
predicted_y = locally_weighted_regression(x.reshape(-1, 1), y, np.array([query_point]), tau)

# Plot the data and prediction
fig, ax = plt.subplots()
ax.plot(x, y, 'o', label='Data')
ax.plot(query_point, predicted_y, 'ro', label='Predicted')
ax.set_title('Locally Weighted Regression')
ax.legend()
st.pyplot(fig)
