# Global Happiness and Welfare Analysis (2005–2021)
## Note:
**Project link is: https://data-assignment1.streamlit.app/**

## Project Description
This project provides a comprehensive, data-driven analysis of global happiness trends and their underlying socioeconomic and institutional drivers between 2005 and 2021. Utilizing a multi-stage visualization strategy implemented in a Streamlit application, the analysis moves from macro-level global context to micro-level component volatility and exceptional case studies.

The core objective is to answer three key questions:

1. What defines the profile of the happiest nations?

2. Which factors contribute most significantly to dynamic changes (increases or declines) in national happiness?

3. What makes some low-income nations "Poor but Happy" exceptions?

## Dataset Details
The analysis is based on the World Happiness Report data.

* Dataset: data.csv (Included in the project folder)

* Source: Helliwell, J. F., Huang, H., Norton, M., & Wang, S. (2024). World Happiness Report (Various Editions, 2005-2021 Data).

## Key Metrics Used:
* *Life Ladder* (Happiness Score)
* *Log GDP per* capita
* *Social support*
* *Healthy life expectancy at birth*
* *Freedom to make life choices*
* *Generosity*
* *Perceptions of corruption*


## Setup
Prerequisites
- Python 3.8+ must be installed.

1. Installation Steps:
```bash
pip install -r requirements.txt
```

2. Run app.py
```bash
streamlit run app.py
```


## Analysis and Visualization Flow

This project follows a nine-stage analytical pipeline supported by clear and progressively complex visualizations.

| **Stage #** | **Chart Type** | **Purpose (One Sentence Summary)** |
|-------------|----------------|------------------------------------|
| **1** |  **Choropleth Map** | Provide a global overview of the geographical distribution of happiness scores. |
| **2** |  **Radar Chart** | Identify the ideal happiness profile by showing the average welfare component scores of the top 20 happiest countries. |
| **3** |  **Parallel Coordinates Plot** | Visualize detailed relationships and score differences among welfare components for the top 20 happiest countries. |
| **4** |  **Horizontal Bar Chart** | Detect countries with the largest year-to-year increases and decreases in happiness scores. |
| **5** |  **Stacked Bar Chart** | Determine which welfare components contributed most to the decline in countries with major decreases in happiness. |
| **6** |  **Animated Scatter Plot** | Observe how the relationship between economic prosperity (GDP) and happiness evolves globally over the years. |
| **7** |  **Treemap** | Establish the starting point for a “Poor but Happy” analysis by showing how low-GDP countries are distributed across continents and happiness categories. |
| **8** |  **Differential Heatmap** | Compare the welfare component profiles of low-GDP continents to highlight what differentiates the “poor but happy” continents. |
| **9** |  **Vertical Box Plot** | Analyze the distribution of welfare components for the 20 least happy countries to identify systemic weaknesses. |

## Team Contributions

| **Member** | **Student ID** | |
|------------|----------------|-------------------------------|
| **Cemal Gürsel Kar** | **2021555032** |  |
| **Orhan Can Çıkman** | **[Student ID]** |  |
| **Mustafa Kayıklık** | **[Student ID]** | |