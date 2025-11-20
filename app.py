import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# --- 0. SAYFA AYARLARI ---
st.set_page_config(
    layout="wide", 
    page_title="Mutluluk ve Refah Analizi",
    initial_sidebar_state="expanded"
)

# --- 1. VERİ YÜKLEME VE ÖN İŞLEME ---

# Veriyi önbelleğe alarak dashboard'u hızlandırır
@st.cache_data
def load_data():
    try:
        # data.csv dosyasının Streamlit uygulamasıyla aynı dizinde olduğunu varsayıyoruz
        df = pd.read_csv('data.csv')
        
        # Gereksiz sütunları temizle
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=[col for col in df.columns if 'Unnamed: 0' in col], errors='ignore')

        # Kıta adı (cont) için eksik verileri kıtasız olarak işaretle
        df['cont'] = df['cont'].fillna('Diğer/Bilinmiyor')
        
        return df
    except FileNotFoundError:
        st.error("HATA: 'data.csv' dosyası bulunamadı. Lütfen dosyanın Streamlit uygulamasıyla aynı dizinde olduğundan emin olun.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()


# --- 2. GÖRSELLEŞTİRME FONKSİYONLARI ---

def viz_1_choropleth(df_input):
    """Görsel 1: Ülkelerin Ortalama Mutluluk Skoru Haritası"""
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

def viz_2_parallel_coords(df_input):
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
    fig.update_layout(height=600, font=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)

def viz_3_change_bar(df_input):
    """Görsel 3: Mutluluk Sıralaması – En Çok Yükselen / Düşen 10 Ülke"""
    pivot = (df_input[df_input['Year'].isin([2005, 2021])]
             .groupby(['Country name', 'Year'])['Life Ladder']
             .mean()
             .unstack('Year')
             .dropna())
    
    pivot['change'] = pivot[2021] - pivot[2005]
    top_last = (pd.concat([pivot.nlargest(10, 'change'), pivot.nsmallest(10, 'change')])
                .rename_axis('country')
                .reset_index()
                .sort_values('change'))
    
    fig = go.Figure()
    colors = ['green' if v > 0 else 'red' for v in top_last['change']]
    
    fig.add_bar(x=top_last['change'],
                y=top_last['country'],
                orientation='h',
                marker_color=colors,
                text=top_last['change'].round(2),
                textposition='outside')
    
    fig.update_layout(
        title='3. 2005 → 2021 Mutluluk Sıralaması – En Çok Yükselen / Düşen 10 Ülke',
        xaxis_title='Life Ladder Değişimi (2021 - 2005)',
        height=550,
        yaxis_autorange='reversed',
        xaxis=dict(tickformat='.2f'))
    
    st.plotly_chart(fig, use_container_width=True)

def viz_4_scatter_regression(df_input, year):
    """Görsel 4: GSYİH vs. Mutluluk (Saçılım ve Regresyon)"""
    df_2021 = df_input[df_input['Year'] == year].dropna(subset=['Log GDP per capita', 'Life Ladder', 'cont'])
    
    fig = px.scatter(
        df_2021,
        x='Log GDP per capita',
        y='Life Ladder',
        hover_name='Country name',
        color='cont',
        trendline='ols', # Regresyon çizgisi
        title=f'4. GSYİH (Logaritmik) vs. Mutluluk Skoru ({year} Yılı) [İLERİ DÜZEY]',
        labels={'Log GDP per capita': 'Logaritmik GSYİH (Kişi Başına)', 'Life Ladder': 'Mutluluk Skoru', 'cont': 'Kıta'},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

def viz_5_heatmap(df_input):
    """Görsel 5: Refah Göstergeleri Arası Korelasyon Matrisi"""
    correlation_cols = ['Life Ladder', 'Log GDP per capita', 'Social support', 
                        'Healthy life expectancy at birth', 'Freedom to make life choices', 
                        'Generosity', 'Perceptions of corruption']
    
    corr_matrix = df_input[correlation_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect='auto',
        color_continuous_scale='RdBu_r', 
        zmin=-1, 
        zmax=1,  
        title='5. Temel Refah Göstergeleri Arasındaki Korelasyon Matrisi (2005-2021) [İLERİ DÜZEY]',
        labels=dict(color="Korelasyon Değeri")
    )
    fig.update_xaxes(tickangle=45, side='bottom')
    fig.update_yaxes(autorange="reversed") 
    st.plotly_chart(fig, use_container_width=True)

def viz_6_line_trend(df_input):
    """Görsel 6: Kıtalara Göre Mutluluk Skoru Trendi"""
    avg_happiness_over_time = df_input.groupby(['Year', 'cont'])['Life Ladder'].mean().reset_index()
    
    fig = px.line(
        avg_happiness_over_time,
        x='Year',
        y='Life Ladder',
        color='cont',
        title='6. Kıtalara Göre Ortalama Mutluluk Skoru Trendi (2005–2021) [İLERİ DÜZEY]',
        labels={'cont': 'Kıta', 'Life Ladder': 'Ortalama Mutluluk Skoru', 'Year': 'Yıl'}
    )
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)

def viz_7_decline_analysis(df_input):
    """Görsel 7: Mutluluk Düşüşü Yaşayanlarda Refah Bileşenlerindeki Değişim"""
    START_YEAR, END_YEAR = 2005, 2021
    metrics_to_examine = ['Log GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']

    df_filtered = df_input[df_input['Year'].isin([START_YEAR, END_YEAR])].copy()
    required_cols = ['Country name', 'Year', 'Life Ladder'] + metrics_to_examine
    df_complete = df_filtered.dropna(subset=required_cols)

    pivot_complete = df_complete.pivot_table(index='Country name', columns='Year', values=required_cols[2:], aggfunc='mean')
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
        title=f'7. Mutluluk Düşüşü Yaşayan İlk 10 Ülkede Refah Bileşenlerindeki ORTALAMA Değişim ({START_YEAR}-{END_YEAR}) [İLERİ DÜZEY]',
        labels={'Ortalama Değişim': 'Ortalama Değişim Skoru (2021 - 2005)'},
        text_auto='.2f'
    )
    fig.update_layout(xaxis_range=[-0.5, 0.5])
    st.plotly_chart(fig, use_container_width=True)

def viz_8_treemap(df_input):
    """Görsel 8: Düşük GDP Ülkeleri Treemap (Nüfus vs Mutluluk)"""
    gdp_40 = df_input['Log GDP per capita'].quantile(0.4)
    low_gdp = df_input[df_input['Log GDP per capita'] <= gdp_40].copy()
    
    plot_df = low_gdp.groupby(['Country name', 'cont']).agg(
        Life_Ladder=('Life Ladder', 'mean'),
        Pop=('Life Ladder', 'count') # Basitçe gözlem sayısını Nüfus temsili olarak alıyoruz.
    ).reset_index()

    fig = px.treemap(plot_df,
                      path=['cont', 'Country name'],
                      values='Pop',
                      color='Life_Ladder',
                      color_continuous_scale='RdYlGn',
                      title='8. Düşük-GDP Ülkeleri – Nüfus (alan) vs Mutluluk (renk) [İLERİ DÜZEY]')
    st.plotly_chart(fig, use_container_width=True)

def viz_9_radar_chart(df_input):
    """Görsel 9: Düşük GDP Ülkelerinde Normalleştirilmiş Kontrast Profilleri"""
    gdp_40_quantile = df_input['Log GDP per capita'].quantile(0.4)
    df_low_gdp = df_input[df_input['Log GDP per capita'] <= gdp_40_quantile].copy()

    df_2021 = df_low_gdp[df_low_gdp['Year'] == 2021].dropna(subset=['Log GDP per capita', 'Social support', 'Freedom to make life choices', 'Perceptions of corruption'])

    # Yoksul Ama Mutlu (Costa Rica) vs Yoksul ve Mutsuz (Afghanistan) kontrastı
    countries_to_compare = ['Costa Rica', 'Afghanistan', 'Senegal'] 

    metrics = ['Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', 
               'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']

    scaler = MinMaxScaler()
    df_2021.loc[:, metrics] = scaler.fit_transform(df_2021[metrics])

    # Yolsuzluk Algısı'nı Şeffaflık olarak ters çevir (1 - skor)
    df_2021.loc[:, 'Perceptions of transparency'] = 1 - df_2021['Perceptions of corruption']
    metrics_radar = metrics.copy()
    metrics_radar[metrics_radar.index('Perceptions of corruption')] = 'Perceptions of transparency'
    
    plot_df = df_2021[df_2021['Country name'].isin(countries_to_compare)]

    normalized_plot_df = plot_df.melt(id_vars='Country name', value_vars=metrics_radar, var_name='Refah Bileşeni', value_name='Normalleştirilmiş Skor')

    fig = px.line_polar(
        normalized_plot_df,
        r='Normalleştirilmiş Skor',
        theta='Refah Bileşeni',
        color='Country name',
        line_close=True,
        title='9. Düşük GDP Ülkelerinde Normalleştirilmiş Refah Profilleri (2021) [İLERİ DÜZEY]',
        labels={'Normalleştirilmiş Skor': 'Göreceli Skor (0-1 Arası)'}
    )

    fig.update_traces(fill='toself', opacity=0.4)
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))), legend_title='Ülke Adı')
    
    st.plotly_chart(fig, use_container_width=True)


# --- 3. DASHBOARD DÜZENİ ---

st.title("Küresel Mutluluk ve Refah Analizi Dashboard'u")
st.markdown("---")

# 9 Buton için seçenekler
VISUALIZATIONS = {
    "1. Global Harita (Life Ladder)": viz_1_choropleth,
    "2. TOP-30 Ülke Profilleri (Paralel Koor.)": viz_2_parallel_coords,
    "3. En Çok Düşüş/Yükseliş (Zaman Serisi)": viz_3_change_bar,
    "4. GDP vs Mutluluk (Regresyon)": viz_4_scatter_regression,
    "5. Tüm Faktörler Arası Korelasyon (Heatmap)": viz_5_heatmap,
    "6. Kıtasal Mutluluk Trendi (Çizgi Grafik)": viz_6_line_trend,
    "7. Düşüş Nedenleri (Ort. Değişim Bar)": viz_7_decline_analysis,
    "8. Düşük GDP Segmenti (Treemap)": viz_8_treemap,
    "9. Yoksul-Mutlu Kontrastı (Radar Grafik)": viz_9_radar_chart
}

# Butonları sidebar'da düzenleyelim
st.sidebar.title("Analiz Seçimi (9 Görsel)")

# st.radio, buton grubunun en temiz Streamlit karşılığıdır
selected_viz = st.sidebar.radio(
    "Lütfen incelemek istediğiniz görseli seçin:",
    list(VISUALIZATIONS.keys()),
    index=0 # Başlangıçta 1. görsel açık olsun
)

# -----------------------------------------------------------
# III. GÖRSELİ ÇALIŞTIRMA VE ETKİLEŞİM UYGULAMA
# -----------------------------------------------------------

# Yıl seçimi slider'ı (Görsel 4 için zorunlu etkileşim)
current_year = df['Year'].max()
if selected_viz == "4. GDP vs Mutluluk (Regresyon)":
    st.subheader(f"Görsel 4 Ayarları (Etkileşim)")
    year_slider = st.slider(
        "Görsel 4'te gösterilecek YIL'ı seçin:",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=current_year,
        step=1
    )
    # Görseli çağır
    VISUALIZATIONS[selected_viz](df, year_slider)
    
elif selected_viz == "6. Kıtasal Mutluluk Trendi (Çizgi Grafik)":
    st.subheader(f"Görsel 6 Ayarları (Etkileşim)")
    # Kıta filtresi ekleyelim (İkinci zorunlu etkileşim)
    all_continents = sorted(df['cont'].dropna().unique().tolist())
    selected_continents = st.multiselect(
        "Görselde Gösterilecek Kıtaları Seçin:",
        options=all_continents,
        default=all_continents
    )
    df_filtered_cont = df[df['cont'].isin(selected_continents)]
    
    # Görseli çağır
    VISUALIZATIONS[selected_viz](df_filtered_cont)

else:
    # Diğer görselleri (Statik Analizler) çağır
    VISUALIZATIONS[selected_viz](df)

st.markdown("---")
st.caption("CEN445 Veri Görselleştirme Projesi | Streamlit, Plotly, Pandas | Tüm görsellerde mouse hover, zoom, pan özellikleri mevcuttur.")