import streamlit as st
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian Network
model = BayesianNetwork([
    ('Weather', 'Rain'),
    ('Sprinkler', 'Rain'),
    ('Rain', 'GrassWet'),
])

# Define the CPDs (Conditional Probability Distributions) for each node
cpd_weather = TabularCPD(variable='Weather', variable_card=2, values=[[0.6], [0.4]])
cpd_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2, values=[[0.5], [0.5]])
cpd_rain = TabularCPD(variable='Rain', variable_card=2,
                      values=[[0.9, 0.7, 0.8, 0.1],
                              [0.1, 0.3, 0.2, 0.9]],
                      evidence=['Weather', 'Sprinkler'], evidence_card=[2, 2])
cpd_grasswet = TabularCPD(variable='GrassWet', variable_card=2,
                          values=[[0.99, 0.1],
                                  [0.01, 0.9]],
                          evidence=['Rain'], evidence_card=[2])

# Add the CPDs to the model
model.add_cpds(cpd_weather, cpd_sprinkler, cpd_rain, cpd_grasswet)

# Validate the model
assert model.check_model()

# Streamlit app setup
st.title('Bayesian Network Inference App')

st.write("### Bayesian Network Structure")
st.graphviz_chart(model.to_daft().render())

# Section to input evidence
st.header('Provide Evidence')
evidence = {}
variables = ['Weather', 'Sprinkler', 'Rain']

for var in variables:
    state = st.radio(f'State of {var}', ['None', 'True', 'False'])
    if state != 'None':
        evidence[var] = 1 if state == 'True' else 0

# Perform inference based on the input evidence
st.header('Inference Results')
inference = VariableElimination(model)

# Determine which variables to query based on the evidence provided
query_vars = [var for var in variables + ['GrassWet'] if var not in evidence]
result = inference.query(query_vars, evidence=evidence)

# Display the probability distributions for the queried variables
for var in query_vars:
    st.write(f"#### Probability Distribution for {var}")
    st.write(result[var])
