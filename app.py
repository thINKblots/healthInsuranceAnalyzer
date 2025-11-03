import streamlit as st
import pandas as pd
import numpy as np
from anthropic import Anthropic
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Health Insurance Analyzer", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('data/insurance.csv')
    return df

if 'messages' not in st.session_state:
    st.session_state.messages = []

def analyze_data(df, user_question, api_key):
    context = f"""
    Dataset Overview:
    - Columns: {df.columns.tolist()}
    - Shape: {df.shape}
    - Sample Data:\n{df.head().to_string()}
    - Data Types:\n{df.dtypes.to_string()}
    """

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""You are a data analyst. Give this dataset:

{context}

User question: {user_question}

Provide a clear answer with insights. If calculations are needed, show them."""
        }]
    )

    return response.content[0].text

st.title("Health Insurance Analyzer")

with st.sidebar:
    api_key = st.text_input("Anthropic API Key", type="password")

df = load_data()

st.write(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.sidebar.checkbox("Show Data Summary"):
    st.write(df.describe())

if st.sidebar.checkbox("Show Correlations"):
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm',
                    center=0, fmt='.2f', ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric columns found for correlation.")

if prompt := st.chat_input("Ask about the health insurance data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            response = "Please enter your API key in the sidebar."
        else:
            response = analyze_data(df, prompt, api_key)

        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
