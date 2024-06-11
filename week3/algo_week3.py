import streamlit as st
import numpy as np

class NeuralNetwork:
    def _init_(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.randn(1, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.random.randn(1, self.output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def feedforward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        return self.sigmoid(self.output)
    
    def backpropagation(self, X, y, learning_rate):
        # Forward pass
        self.feedforward(X)
        
        # Calculate output layer error
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta) * learning_rate
    
    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            self.backpropagation(X, y, learning_rate)

# Create a Streamlit app
def main():
    st.title("Neural Network with Backpropagation")

    # Define input data and labels
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Define neural network parameters
    input_size = 2
    hidden_size = 4
    output_size = 1
    epochs = st.slider("Number of epochs", min_value=100, max_value=10000, step=100, value=1000)
    learning_rate = st.slider("Learning rate", min_value=0.01, max_value=1.0, step=0.01, value=0.1)

    # Create and train neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X, y, epochs, learning_rate)

    # Test the trained neural network
    st.write("Testing the trained neural network:")
    for i in range(len(X)):
        prediction = nn.feedforward(X[i])
        st.write(f"Input: {X[i]}, Predicted Output: {prediction}")

if __name__ == "__main__":
    main()
