import streamlit as st
import pandas as pd

df = pd.read_csv("worldHappiniesReport2005-2021.csv")
st.dataframe(df)