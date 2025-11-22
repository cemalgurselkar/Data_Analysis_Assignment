import streamlit as st
import pandas as pd
from utils import *

st.set_page_config(
    layout="wide", 
    page_title="Happiness and Well-being Analysis.",
    initial_sidebar_state="expanded"
)

st.title("World Happinies Report (2005-2021)")
st.markdown("---")

VISUALIZATIONS = {
    "1. Global Happiness Rates": viz1,
    "2. Welfare Profile of the Top 20 Happiest Countries": viz2,
    "3. Relationship between Welfare Components (Top 20 Happiest Countries)": viz3,
    "4. The Net Change in Happiness for the Top 10 Improvers and Decliners": viz4,
    "5. Component net change for the biggest happiness decliners": viz5,
    "6. The dynamic evolution of Happiness vs. GDP": viz6,
    "7. Global Distribution of Low-Income Countries": viz7,
    "8. Deconstructing the Continental Gap": viz8,
    "9. Identifying Critical Systemic Weaknesses.": viz9
}

st.sidebar.title("Analysis Selection")

selected_viz = st.sidebar.radio(
    "Please select the visualization you want to examine:",
    list(VISUALIZATIONS.keys()),
    index=0
)

current_year = df['Year'].max()
if selected_viz == "4. GDP vs Happinesss":
    st.subheader(f"Visual 4 Settings")
    year_slider = st.slider(
        "Select the YEAR to be displayed in Visual 4:",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=current_year,
        step=1
    )
    VISUALIZATIONS[selected_viz]()
    
elif selected_viz == "6. Continental Happiness Trend":
    st.subheader(f"Visual 6 Settings")
    all_continents = sorted(df['cont'].dropna().unique().tolist())
    selected_continents = st.multiselect(
        "Select Continents to Display in the Visual:",
        options=all_continents,
        default=all_continents
    )
    
    df_filtered_cont = df[df['cont'].isin(selected_continents)]

    VISUALIZATIONS[selected_viz]()

else:
    VISUALIZATIONS[selected_viz]()