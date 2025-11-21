import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler


def viz_1(df_input):

    avg_happiness = df_input.groupby('Country name')['Life Ladder'].mean().reset_index()
    
    fig = px.choropleth(
        avg_happiness,
        locations='Country name',
        locationmode='country names',
        color='Life Ladder',
        hover_name='Country name',
        hover_data={'Life Ladder': ':.2f'},
        title='1. Ülkelerin Ortalama Mutluluk Skoru (2005–2021)',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True))
    st.plotly_chart(fig, use_container_width=True)

def viz_2(df_input):
    """Görsel 2: TOP-30 Mutlu Ülke – Ham Refah Bileşenleri"""
    top30 = df_input[df_input['Year'] == 2021].nlargest(30, 'Life Ladder')
    
    dims = ['Life Ladder', 'Log GDP per capita', 'Social support', 
            'Healthy life expectancy at birth', 'Freedom to make life choices', 
            'Generosity', 'Perceptions of corruption']
    plot_df = top30[['Country name'] + dims].copy()

    fig = px.parallel_coordinates(
            plot_df,
            dimensions=dims,
            color='Life Ladder',
            color_continuous_scale='viridis',
            labels={c: c.replace(' ', '<br>') for c in dims},
            title='2. TOP-30 Mutlu Ülke – Ham Refah Bileşenleri (2021) [İLERİ DÜZEY]'
    )
    fig.update_layout(height=600, font=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)

def viz_3(df):

    df_2021 = df[df['Year'] == 2021].copy()

    countries_to_compare = ['Finland', 'Turkey', 'Afghanistan']
    metrics = [
        'Log GDP per capita',
        'Social support',
        'Healthy life expectancy at birth',
        'Freedom to make life choices',
        'Generosity',
        'Perceptions of corruption'
    ]

    plot_df = df_2021[df_2021['Country name'].isin(countries_to_compare)][['Country name'] + metrics].set_index('Country name')

    scaler = MinMaxScaler()
    df_2021[metrics] = scaler.fit_transform(df_2021[metrics])

    normalized_plot_df = df_2021[df_2021['Country name'].isin(countries_to_compare)][['Country name'] + metrics].melt(
        id_vars='Country name',
        var_name='Bileşen',
        value_name='Normalleştirilmiş Değer'
    )

    fig = px.line_polar(
        normalized_plot_df,
        r='Normalleştirilmiş Değer',
        theta='Bileşen',
        color='Country name',
        line_close=True,
        title='Seçili Ülkelerin Normalleştirilmiş Refah Profilleri (2021)'
    )

    fig.update_traces(fill='toself')    
    st.plotly_chart(fig, use_container_width=True)

def viz_4(df):

    df = pd.read_csv("data.csv")

    correlation_cols = [
        'Life Ladder',
        'Log GDP per capita',
        'Social support',
        'Healthy life expectancy at birth',
        'Freedom to make life choices',
        'Generosity',
        'Perceptions of corruption'
    ]

    corr_matrix = df[correlation_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1, 
        zmax=1,  
        title='Temel Refah Göstergeleri Arasındaki Korelasyon Matrisi (2005-2021)',
        labels=dict(color="Korelasyon Değeri")
    )

    fig.update_xaxes(tickangle=45, side='bottom')
    fig.update_yaxes(autorange="reversed") 

    st.plotly_chart(fig, use_container_width=True)

def viz_5(df):

    df = pd.read_csv('data.csv')

    pivot = (df[df['Year'].isin([2005, 2021])]
            .groupby(['Country name', 'Year'])['Life Ladder']
            .mean()
            .unstack('Year')
            .dropna())

    pivot['change'] = pivot[2021] - pivot[2005]

    top10  = pivot.nlargest(10, 'change')
    last10 = pivot.nsmallest(10, 'change')
    plot_df = (pd.concat([top10, last10])
            .rename_axis('country')
            .reset_index()
            .sort_values('change'))

    fig = go.Figure()
    colors = ['green' if v > 0 else 'red' for v in plot_df['change']]
    fig.add_bar(x=plot_df['change'],
                y=plot_df['country'],
                orientation='h',
                marker_color=colors,
                text=plot_df['change'].round(2),
                textposition='outside')

    fig.update_layout(
        title='2005 → 2021 Mutluluk Sıralaması – En Çok Yükselen / Düşen 10 Ülke',
        xaxis_title='Life Ladder Değişimi (2021 - 2005)',
        height=450,
        yaxis_autorange='reversed',
        xaxis=dict(tickformat='.2f'))
    st.plotly_chart(fig, use_container_width=True)

def viz_6(df):

    df = pd.read_csv("data.csv")

    START_YEAR = 2005 
    END_YEAR = 2021

    metrics_to_examine = [
        'Log GDP per capita',
        'Social support', 
        'Freedom to make life choices',
        'Generosity',
        'Perceptions of corruption'
    ]

    df_filtered = df[df['Year'].isin([START_YEAR, END_YEAR])].copy()

    required_cols = ['Country name', 'Year', 'Life Ladder'] + metrics_to_examine
    df_complete = df_filtered.dropna(subset=required_cols)

    pivot_complete = df_complete.pivot_table(
        index='Country name', 
        columns='Year', 
        values=required_cols[2:],
        aggfunc='mean'
    )

    pivot_complete['Life Ladder Change'] = pivot_complete[('Life Ladder', END_YEAR)] - pivot_complete[('Life Ladder', START_YEAR)]

    top_declines_series = pivot_complete['Life Ladder Change'].nsmallest(10)
    decline_countries = top_declines_series.index.tolist()

    change_df = pivot_complete.loc[decline_countries].copy()

    average_changes = {}
    for metric in metrics_to_examine:
        change_values = change_df[(metric, END_YEAR)] - change_df[(metric, START_YEAR)]
        average_changes[metric] = change_values.mean()

    plot_data = pd.DataFrame(list(average_changes.items()), columns=['Refah Bileşeni', 'Ortalama Değişim'])

    plot_data['Durum'] = plot_data['Ortalama Değişim'].apply(lambda x: 'İyileşme (Artış)' if x >= 0 else 'Kötüleşme (Düşüş)')

    fig = px.bar(
        plot_data.sort_values(by='Ortalama Değişim', ascending=True),
        x='Ortalama Değişim',
        y='Refah Bileşeni',
        orientation='h',
        color='Durum',
        color_discrete_map={'İyileşme (Artış)': 'green', 'Kötüleşme (Düşüş)': 'red'},
        title=f'Mutluluk Düşüşü Yaşayan İlk 10 Ülkede Refah Bileşenlerindeki ORTALAMA Değişim ({START_YEAR}-{END_YEAR})',
        labels={'Ortalama Değişim': 'Ortalama Değişim Skoru (2021 - 2005)'},
        text_auto='.2f'
    )

    fig.update_layout(
        xaxis_title='Ortalama Değişim Skoru',
        yaxis_title='Refah Bileşeni',
        xaxis_range=[-0.5, 0.5]
    )

    st.plotly_chart(fig, use_container_width=True)

def viz_7(df):

    df = pd.read_csv('data.csv')

    gdp_40 = df['Log GDP per capita'].quantile(0.4)
    low_gdp = df[df['Log GDP per capita'] <= gdp_40].copy()

    plot_df = low_gdp.groupby(['Country name', 'cont']).agg(
        Life_Ladder=('Life Ladder', 'mean'),
        Pop=('Life Ladder', 'count')
    ).reset_index()

    plot_df['Mutluluk'] = pd.cut(plot_df['Life_Ladder'],
                                bins=[0, 4.5, 6.5, 10],
                                labels=['Düşük', 'Orta', 'Yüksek'])

    fig = px.treemap(plot_df,
                    path=['cont', 'Country name'],
                    values='Pop',
                    color='Life_Ladder',
                    color_continuous_scale='RdYlGn',
                    title='Düşük-GDP Ülkeleri – Nüfus (alan) vs Mutluluk (renk)')
    st.plotly_chart(fig, use_container_width=True)

def viz_8(df):

    df = pd.read_csv('data.csv')
    gdp_40 = df['Log GDP per capita'].quantile(0.4)
    low = df[df['Log GDP per capita'] <= gdp_40].copy()

    cols = ['Healthy life expectancy at birth','Social support',
            'Freedom to make life choices','Generosity','Perceptions of corruption']
    low[cols] = low[cols].clip(0, 1)
    low['Clean'] = 1 - low['Perceptions of corruption']

    agg = low.groupby('cont')[cols[:-1] + ['Clean']].mean()
    agg = agg[agg.index.isin(['AF','SA','EU'])]

    ref = agg.loc['AF']
    diff = agg.loc[['SA','EU']] - ref

    diff_df = (diff.reset_index()
                    .melt(id_vars='cont', var_name='Bilesen', value_name='Fark'))

    fig = go.Figure()
    for kita in ['SA','EU']:
        df_ = diff_df[diff_df['cont']==kita]
        fig.add_trace(go.Bar(
                y=df_['Bilesen'],
                x=df_['Fark'],
                name=kita,
                orientation='h',
                base=0,
                text=df_['Fark'].round(2),
                textposition='outside'
        ))

    fig.update_layout(
            title='Düşük-GDP Kıtaları – AF Referansına Göre Fark (SA & EU)',
            xaxis_title='Fark (SA veya EU ortalama − AF ortalama)',
            yaxis_title=None,
            height=400,
            bargap=0.2,
            legend_title='Kıta',
            xaxis=dict(tickformat='.2f')
    )

    st.plotly_chart(fig, use_container_width=True)

def viz_9(df):
    """Görsel 9: En Düşük GDP'li 5 Ülkede Normalleştirilmiş Kontrast Profilleri (2021)"""

    # Dosya okuma işlemi
    df = pd.read_csv("data.csv") 

    gdp_40_quantile = df['Log GDP per capita'].quantile(0.4)
    df_low_gdp = df[df['Log GDP per capita'] <= gdp_40_quantile].copy()

    # NaN değerleri olanları eleyip 2021 verisini alıyoruz
    df_2021 = df_low_gdp[df_low_gdp['Year'] == 2021].dropna(subset=['Log GDP per capita', 'Social support', 'Freedom to make life choices', 'Perceptions of corruption'])

    # 🎯 YENİ LİSTE: NaN içermeyen düşük GDP grubundan, GDP'si en düşük 5 ülkeyi alıyoruz
    top5_lowest_gdp = df_2021.nsmallest(5, 'Log GDP per capita')
    countries_to_compare = top5_lowest_gdp['Country name'].tolist()
    
    # Konsol çıktısı için (Streamlit'te görünmez, ancak hata ayıklamaya yardımcı olur)
    # print(f"Kıyaslanan Ülkeler: {countries_to_compare}") 

    metrics = [
        'Log GDP per capita',
        'Social support',
        'Healthy life expectancy at birth',
        'Freedom to make life choices',
        'Generosity',
        'Perceptions of corruption'
    ]

    scaler = MinMaxScaler()
    df_2021.loc[:, metrics] = scaler.fit_transform(df_2021[metrics])

    # Şeffaflık Algısı dönüşümü
    df_2021.loc[:, 'Perceptions of transparency'] = 1 - df_2021['Perceptions of corruption']
    metrics = [m if m != 'Perceptions of corruption' else 'Perceptions of transparency' for m in metrics]

    plot_df = df_2021[df_2021['Country name'].isin(countries_to_compare)]

    normalized_plot_df = plot_df.melt(
        id_vars='Country name',
        value_vars=metrics,
        var_name='Refah Bileşeni',
        value_name='Normalleştirilmiş Skor'
    )

    # 5 ülkeye uygun, zıt ve koyu renk paleti (Plotly D3 serisinden)
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 

    # GRAFİK TÜRÜ: GRUP ÇUBUK GRAFİĞİ
    fig = px.bar(
        normalized_plot_df,
        x='Refah Bileşeni',
        y='Normalleştirilmiş Skor',
        color='Country name',
        barmode='group', 
        title=f'En Düşük GDP\'ye Sahip 5 Ülkenin Bileşen Bazında Kontrast Profilleri (2021) - ({", ".join(countries_to_compare)})',
        labels={'Normalleştirilmiş Skor': 'Göreceli Skor (0-1 Arası)'},
        color_discrete_sequence=custom_colors,
        text='Normalleştirilmiş Skor' 
    )

    fig.update_xaxes(tickangle=45)
    
    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    
    fig.update_yaxes(range=[0, 1.1], title_text='Normalleştirilmiş Skor') 

    st.plotly_chart(fig, use_container_width=True)