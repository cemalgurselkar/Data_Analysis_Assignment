import streamlit as st
import pandas as pd
from utils import *

st.set_page_config(
    layout="wide", 
    page_title="Happiness and Well-being Analysis.",
    initial_sidebar_state="expanded"
)

def load_data():
    df = pd.read_csv("data.csv")
    df = df.drop(columns=["Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0"], axis=1)
    return df

df = load_data()

st.title("World Happinies Report (2005-2021)")
st.markdown("---")

VISUALIZATIONS = {
    "1. Global Map": viz_1,
    "2. Components of prosperity.": viz_2,
    "3. Normalized Welfare Profiles of Selected Countries.": viz_3,
    "4. Correlation Matrix Among Basic Welfare Indicators.": viz_4,
    "5. Countries with the largest declines and increases in happiness rates between 2005 and 2021. ": viz_5,
    "6. Average Change in Welfare Components in the Top 10 Countries Experiencing a Decline in Happiness.": viz_6,
    "7. Low-GDP Countries – Population (area) vs Happiness": viz_7,
    "8. Low-GDP Regions – Difference According to AF Reference (SA & EU)": viz_8,
    "9. Normalized Welfare Profiles in Low GDP Countries (2021)": viz_9
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
    VISUALIZATIONS[selected_viz](df, year_slider)
    
elif selected_viz == "6. Continental Happiness Trend":
    st.subheader(f"Visual 6 Settings")
    all_continents = sorted(df['cont'].dropna().unique().tolist())
    selected_continents = st.multiselect(
        "Select Continents to Display in the Visual:",
        options=all_continents,
        default=all_continents
    )
    
    df_filtered_cont = df[df['cont'].isin(selected_continents)]

    VISUALIZATIONS[selected_viz](df_filtered_cont)

else:
    VISUALIZATIONS[selected_viz](df)