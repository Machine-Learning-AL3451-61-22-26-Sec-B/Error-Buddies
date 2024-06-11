import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def candidate_elimination(data):
    # Separating concept features from Target
    concepts = np.array(data.iloc[:,0:-1])
    # Isolating target into a separate array
    target = np.array(data.iloc[:,-1])

    # Initialise S0 with the first instance from concepts
    specific_h = concepts[0].copy()
    # Initialize general_h with the same shape as specific_h
    general_h = [['?' for _ in specific_h] for _ in specific_h]

    # The learning iterations
    for i, h in enumerate(concepts):
        # Checking if the hypothesis has a positive target
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        # Checking if the hypothesis has a negative target
        if target[i] == "No":
            for x in range(len(specific_h)):
                if specific_h[x] != h[x] and specific_h[x] != '?':
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    # Filter out rows in general_h that are fully '?'
    general_h = [hypothesis for hypothesis in general_h if not all(attr == '?' for attr in hypothesis)]
    return specific_h, general_h

def main():
    st.title("Candidate Elimination Algorithm")

    st.write("Upload your dataset (CSV format):")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Original Dataset:")
        st.dataframe(data.style.set_table_attributes('style="font-size: 14px; background-color: #f2f2f2"'))

        specific_h, general_h = candidate_elimination(data)

        st.write("Final Specific Hypothesis:")
        st.dataframe(pd.DataFrame(specific_h).T.style.set_table_attributes('style="font-size: 14px; background-color: #f2f2f2"'))

        st.write("Final General Hypotheses:")
        st.write("Number of General Hypotheses:", len(general_h))
        if len(general_h) > 0:
            st.write("General Hypotheses:")
            for i, hypothesis in enumerate(general_h, start=1):
                st.write(f"Hypothesis {i}:")
                st.dataframe(pd.DataFrame(hypothesis).T.style.set_table_attributes('style="font-size: 14px; background-color: #f2f2f2"'))

if __name__ == "__main__":
    main()
