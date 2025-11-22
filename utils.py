import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

def load_data():
    return pd.read_csv("data.csv")

df = load_data()

components = [
    'Life Ladder',
    'Log GDP per capita',
    'Social support',
    'Healthy life expectancy at birth',
    'Freedom to make life choices',
    'Generosity',
    'Perceptions of corruption'
]

def viz1():
    avg_happiness = df.groupby('Country name')['Life Ladder'].mean().reset_index()

    fig = px.choropleth(
        avg_happiness,
        locations='Country name',
        locationmode='country names',
        color='Life Ladder',
        hover_name='Country name',
        hover_data={'Life Ladder': ':.2f'},
        title="Countries Average Happiness Score (2005–2021)",
        color_continuous_scale='Viridis'
    )

    fig.update_layout(geo=dict(showframe=False, showcoastlines=True))
    st.plotly_chart(fig)

def viz2():
    radar_components = [
        'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
        'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
    ]
    
    avg_df = df.groupby('Country name')[['Life Ladder'] + radar_components].mean().dropna()
    top_20_df = avg_df.sort_values(by='Life Ladder', ascending=False).head(20)

    df_norm = top_20_df[radar_components].copy()
    df_norm['Perceptions of corruption'] = 1 - df_norm['Perceptions of corruption']

    scaler = MinMaxScaler()
    df_norm_scaled = pd.DataFrame(scaler.fit_transform(df_norm), columns=df_norm.columns)

    avg_profile = df_norm_scaled.mean().reset_index()
    avg_profile.columns = ['Component', 'Score']

    component_labels = {
        'Log GDP per capita': 'GDP',
        'Social support': 'Social Support',
        'Healthy life expectancy at birth': 'Health Expectancy',
        'Freedom to make life choices': 'Freedom',
        'Generosity': 'Generosity',
        'Perceptions of corruption': 'Low Corruption'
    }
    avg_profile['Component'] = avg_profile['Component'].map(component_labels)

    fig = px.line_polar(
        avg_profile, r='Score', theta='Component',
        line_close=True,
        title='Welfare Profile (Average Normalized Scores) of the Top 20 Happiest Countries',
    )
    fig.update_traces(fill='toself')
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), title_x=0.5)
    st.plotly_chart(fig)

def viz3():
    components_to_plot = components.copy()

    avg_df = df.groupby('Country name')[components_to_plot].mean().dropna()
    df_plot = avg_df.sort_values(by='Life Ladder', ascending=False).head(20)

    dim_labels = {
        'Life Ladder': 'Happiness Score',
        'Log GDP per capita': 'GDP',
        'Social support': 'Social-Support',
        'Healthy life expectancy at birth': 'Health-Expectancy',
        'Freedom to make life choices': 'Freedom',
        'Generosity': 'Generosity',
        'Perceptions of corruption': 'Corruption'
    }

    fig = px.parallel_coordinates(
        df_plot,
        dimensions=components_to_plot,
        color='Life Ladder',
        labels=dim_labels,
        color_continuous_scale=px.colors.sequential.Plasma,
    )

    fig.update_layout(width=1200, height=600, margin=dict(l=100, r=100, t=50, b=50))
    st.plotly_chart(fig, config= {'displayModeBar': False})

def viz4():
    df_sorted = df.sort_values('Year')
    df_first = df_sorted.groupby('Country name').first()
    df_last = df_sorted.groupby('Country name').last()

    df_delta = (df_last['Life Ladder'] - df_first['Life Ladder']).reset_index()
    df_delta.columns = ['Country name', 'Delta Life Ladder']
    df_delta = df_delta.dropna()

    top_improvers = df_delta.nlargest(10, 'Delta Life Ladder')
    top_decliners = df_delta.nsmallest(10, 'Delta Life Ladder')

    df_movers = pd.concat([top_decliners, top_improvers]).sort_values('Delta Life Ladder')
    df_movers['Change Type'] = df_movers['Delta Life Ladder'].apply(lambda x: 'Increase' if x > 0 else 'Decline')

    fig = px.bar(
        df_movers,
        x='Delta Life Ladder',
        y='Country name',
        color='Change Type',
        orientation='h',
        title='Net Change in Happiness (Life Ladder) for Top Movers (2005-2021)',
        color_discrete_map={'Increase': 'green', 'Decline': 'red'}
    )
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig)

def viz5():
    df_sorted = df.sort_values('Year')
    df_first = df_sorted.groupby('Country name').first()
    df_last = df_sorted.groupby('Country name').last()

    df_delta = df_last[components] - df_first[components]
    df_delta['Country name'] = df_last.index
    df_delta = df_delta.dropna()

    decliners = df_delta.nsmallest(10, 'Life Ladder')

    delta_components = [c for c in df_delta.columns if c.startswith('Log') or c in components[2:]]

    df_melt = decliners.melt(
        id_vars='Country name',
        value_vars=[f'{c}' for c in delta_components if c != 'Life Ladder'],
        var_name='Welfare Component',
        value_name='Change Value'
    )

    rename_map = {
        'Log GDP per capita': 'Log GDP',
        'Social support': 'Social Support',
        'Healthy life expectancy at birth': 'Health Expectancy',
        'Freedom to make life choices': 'Freedom',
        'Generosity': 'Generosity',
        'Perceptions of corruption': 'Corruption Perception'
    }
    df_melt['Welfare Component'] = df_melt['Welfare Component'].map(rename_map)

    fig = px.bar(
        df_melt,
        x='Country name',
        y='Change Value',
        color='Welfare Component',
        title='Component Net Change for the 10 Biggest Happiness Decliners',
    )
    fig.update_layout(barmode='relative', title_x=0.5)
    st.plotly_chart(fig)

def viz6():
    df_anim = df.dropna(subset=['cont']).copy()

    avg_life = df_anim.groupby('Country name')['Life Ladder'].mean()
    df_anim['Life Ladder_size'] = df_anim['Country name'].map(avg_life)

    fig = px.scatter(
        df_anim,
        x="Log GDP per capita",
        y="Life Ladder",
        animation_frame="Year",
        animation_group="Country name",
        size="Life Ladder_size",
        color="cont",
        hover_name="Country name",
        title="Global Evolution of Happiness vs. GDP (2005-2021)",
    )

    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig)

def viz7():
    gdp_40 = df['Log GDP per capita'].quantile(0.4)
    low_gdp = df[df['Log GDP per capita'] <= gdp_40]

    plot_df = low_gdp.groupby(['Country name', 'cont']).agg(
        Life_Ladder=('Life Ladder', 'mean'),
        Pop=('Life Ladder', 'count')
    ).reset_index()

    fig = px.treemap(
        plot_df,
        path=['cont', 'Country name'],
        values='Pop',
        color='Life_Ladder',
        color_continuous_scale='RdYlGn',
        title='Low-GDP Countries'
    )
    st.plotly_chart(fig)

def viz8():
    comparison_components = [
        'Social support', 'Healthy life expectancy at birth',
        'Freedom to make life choices', 'Generosity',
        'Perceptions of corruption'
    ]

    avg_df_country = df.groupby('Country name')[['Life Ladder', 'Log GDP per capita'] + comparison_components] \
                        .mean().reset_index().dropna()

    continent_map = df[['Country name', 'cont']].drop_duplicates().dropna()
    avg_df_country = avg_df_country.merge(continent_map, on='Country name', how='left').dropna()

    gdp_40 = avg_df_country['Log GDP per capita'].quantile(0.4)
    df_low = avg_df_country[avg_df_country['Log GDP per capita'] <= gdp_40]

    if df_low.empty:
        st.warning("Not enough data.")
        return

    continent_profile = df_low.groupby('cont')[comparison_components].mean()
    continent_profile['Perceptions of corruption'] = 1 - continent_profile['Perceptions of corruption']

    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(
        scaler.fit_transform(continent_profile),
        columns=continent_profile.columns,
        index=continent_profile.index
    )

    df_norm = df_norm.rename(columns={
        'Social support': 'Social Support',
        'Healthy life expectancy at birth': 'Health Expectancy',
        'Freedom to make life choices': 'Freedom',
        'Generosity': 'Generosity',
        'Perceptions of corruption': 'Low Corruption'
    })

    heatmap_matrix = df_norm.T

    fig = px.imshow(
        heatmap_matrix,
        color_continuous_scale=px.colors.sequential.YlGnBu,
        labels=dict(x="Continent", y="Welfare Component", color="Score"),
    )

    # --- ANNOTATION EKLE (Karelere değer yazdır)
    fig.update_traces(
        text=heatmap_matrix.round(2),
        texttemplate="%{text}",
        textfont={"size": 14},
    )

    # --- BÜYÜKLÜK AYARLA
    fig.update_layout(
        title="Profile Heatmap of Low GDP Continents",
        width=1000,
        height=700,
        margin=dict(l=100, r=100, t=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)


def viz9():
    components_to_plot = components.copy()

    avg_df = df.groupby('Country name')[components_to_plot].mean().dropna()
    bottom_20 = avg_df.nsmallest(20, 'Life Ladder')

    df_melt = bottom_20.melt(
        value_vars=components_to_plot,
        var_name='Welfare Component',
        value_name='Score'
    )

    norm_factors = df[components_to_plot].agg(['min', 'max'])

    df_melt['Normalized Score'] = df_melt.apply(
        lambda row: (row['Score'] - norm_factors.loc['min', row['Welfare Component']]) /
                    (norm_factors.loc['max', row['Welfare Component']] - norm_factors.loc['min', row['Welfare Component']]),
        axis=1
    )

    df_melt.loc[df_melt['Welfare Component'] == 'Perceptions of corruption', 'Normalized Score'] = \
        1 - df_melt.loc[df_melt['Welfare Component'] == 'Perceptions of corruption', 'Normalized Score']

    df_melt['Welfare Component'] = df_melt['Welfare Component'].map({
        'Life Ladder': 'Happiness Score',
        'Log GDP per capita': 'Log GDP',
        'Social support': 'Social Support',
        'Healthy life expectancy at birth': 'Health Expectancy',
        'Freedom to make life choices': 'Freedom',
        'Generosity': 'Generosity',
        'Perceptions of corruption': 'Low Corruption'
    })

    fig = px.box(
        df_melt,
        x='Welfare Component',
        y='Normalized Score',
        title='Viz 9: Distribution of Normalized Scores for the Bottom 20 Unhappiest Countries'
    )

    fig.update_layout(title_x=0.5, showlegend=False)
    st.plotly_chart(fig)
