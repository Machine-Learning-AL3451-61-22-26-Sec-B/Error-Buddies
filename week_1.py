import streamlit as st
import pandas as pd

def learn(concepts, target):
    # Initialise S0 with the first instance from concepts
    specific_h = concepts[0].copy()

    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    # The learning iterations
    for i, h in enumerate(concepts):

        # Checking if the hypothesis has a positive target
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                # Change values in S & G only if values change
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        # Checking if the hypothesis has a negative target
        if target[i] == "No":
            for x in range(len(specific_h)):
                # For negative hypothesis change values only in G
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    # find indices where we have empty rows, meaning those that are unchanged
    indices = [i for i, val in enumerate(general_h) if val == ['?'] * len(specific_h)]
    for i in indices:
        # remove those rows from general_h
        general_h.remove(['?'] * len(specific_h))

    # Return final values
    return specific_h, general_h

def main():
    st.title("Candidate Elimination Algorithm")

    # Allow users to upload a CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write("Uploaded data:")
        st.write(data)

        # Extract features and target from the uploaded data
        concepts = data.iloc[:, :-1].values.tolist()
        target = data.iloc[:, -1].values.tolist()

        if st.button("Run Algorithm"):
            specific, general = learn(concepts, target)
            st.write("\nFinal Specific Hypothesis:", specific)
            st.write("Final General Hypotheses:")
            for hypothesis in general:
                st.write(hypothesis)

if __name__ == "__main__":
    main()


