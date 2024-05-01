import streamlit as st
import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, value=None, result=None):
        self.feature = feature
        self.value = value
        self.result = result
        self.children = {}

class DecisionTreeID3:
    def __init__(self):
        self.root = None

    def entropy(self, data):
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        return -np.sum(probabilities * np.log2(probabilities))

    def information_gain(self, data, feature_name, target_name):
        total_entropy = self.entropy(data[target_name])
        unique_values = data[feature_name].unique()
        weighted_entropy = 0
        for value in unique_values:
            subset = data[data[feature_name] == value]
            weighted_entropy += len(subset) / len(data) * self.entropy(subset[target_name])
        return total_entropy - weighted_entropy

    def build_tree(self, data, features, target_name):
        if len(data) == 0:
            return None
        if len(data[target_name].unique()) == 1:
            return Node(result=data[target_name].iloc[0])

        information_gains = [(feature, self.information_gain(data, feature, target_name)) for feature in features]
        best_feature, _ = max(information_gains, key=lambda x: x[1])

        root = Node(feature=best_feature)

        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            root.children[value] = self.build_tree(subset, [f for f in features if f != best_feature], target_name)

        return root

    def fit(self, data, target_name):
        features = [col for col in data.columns if col != target_name]
        self.root = self.build_tree(data, features, target_name)

    def predict_instance(self, instance, node):
        if node.result is not None:
            return node.result
        value = instance[node.feature]
        if value not in node.children:
            return None
        return self.predict_instance(instance, node.children[value])

    def predict(self, data):
        predictions = []
        for index, row in data.iterrows():
            result = self.predict_instance(row, self.root)
            predictions.append(result)
        return predictions

def main():
    st.title("Decision Tree ID3 with Streamlit")

    # Upload dataset file
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])
    if uploaded_file is not None:
        # Read the dataset
        data = pd.read_csv(uploaded_file)

        # Check if 'PlayTennis' column exists in the dataset
        if 'PlayTennis' not in data.columns:
            st.error("Target column 'PlayTennis' not found in the uploaded dataset.")
            return

        # Initialize the DecisionTreeID3 model
        model = DecisionTreeID3()

        # Train the model
        model.fit(data, 'PlayTennis')

        # Make predictions
        predictions = model.predict(data)

        st.write("Predictions:", predictions)

if __name__ == "__main__":
    main()
