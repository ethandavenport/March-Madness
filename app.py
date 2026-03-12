import streamlit as st
import pandas as pd

st.title("2025 March Madness Bracket")

bracket = pd.read_csv("bracket_2025.csv")

st.dataframe(bracket)