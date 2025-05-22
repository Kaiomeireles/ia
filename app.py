import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import plotly.express as px
import pydeck as pdk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Configuração da página
st.set_page_config(
    page_title="Impacto da IA no Mercado de Trabalho",
    page_icon="🤖",
    layout="wide"
)

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("dados.csv", encoding="utf-8-sig")
        df.columns = ['Setor', 'Região', 'Impacto']
        df['Setor'] = df['Setor'].astype(str).str.strip()
        df['Região'] = df['Região'].astype(str).str.strip()
        df['Impacto'] = pd.to_numeric(df['Impacto'], errors='coerce')
        return df.dropna()
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

df = carregar_dados()

# Menu lateral
menu = st.sidebar.selectbox(
    "Navegação",
    ["Introdução", "Base de Dados", "Análise Descritiva", "Inferência Estatística", 
     "Regressão Linear", "IA Integrada", "Mapa Geoespacial"]
)
st.sidebar.subheader("Integrantes 👥")
st.sidebar.markdown("""
- **Kaio Vinicius Meireles Alves** — RM553282  
- **Lucas Alves de Souza** — RM553956  
- **Lucas de Freitas Pagung** — RM553242  
- **Guilherme Fernandes de Freitas** — RM554323  
- **João Pedro Chizzolini de Freitas** — RM553172  
""")

# Introdução
if menu == "Introdução":
    st.title("Impacto da IA no Mercado de Trabalho 🚀")
    st.write("""
    Este projeto explora de forma **abrangente** como a **Inteligência Artificial (IA)** está transformando o mercado de trabalho em diferentes setores e regiões do mundo.
    
    A seguir, você encontrará análises e visualizações baseadas em dados, abordando desde estatísticas descritivas até inferências estatísticas, além de uma integração com IA para interação dinâmica.
    """)
    st.subheader("✅ O que você vai ver neste projeto:")
    st.markdown("""
    - **📊 Base de Dados**: Visualização dos dados utilizados, separados por setor e região, com cálculo de intervalos de confiança para estimar a média de impacto da IA.
    - **📈 Análise Descritiva**: Estatísticas básicas e gráficos para entender a distribuição do impacto da IA em diferentes setores.
    - **📉 Inferência Estatística**: Testes de hipóteses para identificar se há diferenças significativas entre os setores quanto ao impacto da IA.
    - **📊 Regressão Linear**: Modelagem estatística para analisar a relação entre variáveis, explorando a correlação entre regiões e impacto da IA.
    - **🤖 IA Integrada**: Um espaço interativo para perguntas, simulando a integração de uma IA no contexto do projeto.
    - **🌍 Mapa Geoespacial**: Apresentação da ideia de mapear o impacto da IA pelo mundo, destacando a importância da geolocalização na análise.
    """)
    st.subheader("🎯 Objetivo Geral")
    st.write("""
    Analisar o **impacto da adoção da IA** nos setores econômicos e nas regiões, utilizando técnicas estatísticas e visualizações, para compreender:
    - Quais setores são mais impactados pela automação.
    - Como o impacto varia entre regiões.
    - Quais tendências podem ser observadas para o futuro do mercado de trabalho.
    """)
    st.subheader("💡 Hipóteses Investigadas")
    st.write("""
    - O setor de **Tecnologia** é o mais impactado positivamente pela IA.
    - Regiões mais **desenvolvidas** possuem maior adoção da automação.
    - Profissões **criativas** tendem a sofrer menos com a automação.
    """)
    st.success("Explore o menu lateral para navegar pelas seções e confira as análises que realizamos! 🚀")

# Base de Dados + Intervalo de Confiança
elif menu == "Base de Dados":
    st.title("Base de Dados 📊")
    if not df.empty:
        st.dataframe(df)
        setor_escolhido = st.selectbox("Selecione o setor:", df['Setor'].unique())
        df_filtrado = df[df['Setor'] == setor_escolhido]
        dados = df_filtrado['Impacto']
        media = np.mean(dados)
        desvio = np.std(dados, ddof=1)
        n = len(dados)
        t_critico = stats.t.ppf(0.975, df=n-1)
        margem_erro = t_critico * (desvio / np.sqrt(n))
        ic_min = media - margem_erro
        ic_max = media + margem_erro
        st.metric("Média de Impacto", f"{media:.2f}%")
        st.write(f"Intervalo de Confiança (95%): **[{ic_min:.2f}%, {ic_max:.2f}%]**")
        fig, ax = plt.subplots()
        ax.bar(setor_escolhido, media, yerr=margem_erro, capsize=10, color='royalblue')
        ax.set_ylabel('Impacto (%)')
        ax.set_title(f'Intervalo de Confiança (95%) - {setor_escolhido}')
        ax.set_ylim(0, 100)
        st.pyplot(fig)
    else:
        st.warning("Dados não carregados.")

# Análise Descritiva
elif menu == "Análise Descritiva":
    st.title("Análise Descritiva 📈")
    st.write("Estatísticas básicas e gráficos.")
    st.write(df.describe())
    st.subheader("Gráfico de Impacto por Setor")
    fig = px.box(df, x="Setor", y="Impacto", color="Setor", title="Distribuição do Impacto por Setor")
    st.plotly_chart(fig)

# Inferência Estatística
elif menu == "Inferência Estatística":
    st.title("Inferência Estatística 📉")
    st.write("Testamos a hipótese de que o setor de Tecnologia possui maior impacto que a média.")
    tecnologia = df[df['Setor'] == 'Tecnologia']['Impacto']
    outros = df[df['Setor'] != 'Tecnologia']['Impacto']
    t_stat, p_valor = stats.ttest_ind(tecnologia, outros)
    st.write(f"T-Stat: {t_stat:.2f}, P-Valor: {p_valor:.4f}")
    if p_valor < 0.05:
        st.success("Resultado significativo: O impacto da IA na Tecnologia é maior que nos outros setores.")
    else:
        st.warning("Sem evidências suficientes para afirmar que o impacto da IA na Tecnologia é maior.")

# Regressão Linear
elif menu == "Regressão Linear":
    st.title("Regressão Linear 📊")
    st.write("Analisamos a relação entre Índice numérico e Impacto.")
    df['Regiao_Num'] = pd.factorize(df['Região'])[0]
    X = df[['Regiao_Num']]
    y = df['Impacto']
    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    r2 = r2_score(y, y_pred)
    st.write(f"Coeficiente de Determinação (R²): {r2:.2f}")
    st.write(f"Equação: Impacto = {modelo.coef_[0]:.2f} * Região_Num + {modelo.intercept_:.2f}")
    fig, ax = plt.subplots()
    ax.scatter(df['Regiao_Num'], df['Impacto'], color='blue', label='Dados')
    ax.plot(df['Regiao_Num'], y_pred, color='red', label='Regressão Linear')
    ax.set_xlabel('Região (Numérica)')
    ax.set_ylabel('Impacto (%)')
    ax.legend()
    st.pyplot(fig)


# IA Integrada
elif menu == "IA Integrada":
    st.title("IA Integrada com OpenRouter 🤖")
    st.write("Interaja com uma IA real via OpenRouter sobre o impacto da automação no mercado de trabalho.")

    pergunta = st.text_input("Faça sua pergunta:")

    if pergunta:
        with st.spinner("Consultando a IA via OpenRouter..."):

            # Carrega e limpa a chave
            api_key = st.secrets["OPENROUTER_API_KEY"].strip()

            # Testes visuais para garantir que a chave está correta
            st.write(f"Chave carregada: {api_key[:8]}********")
            st.write(f"Chave tamanho: {len(api_key)}")  # Deve ser ~66

            headers = {
                "Authorization": f"Bearer {api_key}",  # ATENÇÃO: "Bearer" com espaço!
                "Content-Type": "application/json"
            }

            # Teste: exibe o header (parcial) só para confirmar
            st.write(f"Header Authorization: {headers['Authorization'][:20]}********")

            data = {
                "model": "mistralai/devstral-small:free",  # Modelo gratuito
                "messages": [
                    {"role": "system", "content": "Você é um especialista em mercado de trabalho e inteligência artificial."},
                    {"role": "user", "content": pergunta}
                ]
            }

            # URL correta da API
            url = "https://openrouter.ai/api/v1/chat/completions"

            response = requests.post(
                url,
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                resposta = response.json()
                conteudo = resposta['choices'][0]['message']['content']
                st.success(conteudo)
            else:
                st.error(f"Erro na API: {response.status_code} - {response.text}")

elif menu == "Mapa Geoespacial":
    st.title("Mapa Geoespacial 🌍")
    st.write("Visualize o impacto da IA em diferentes regiões do mundo com um mapa interativo.")

    # Dados fictícios representando o impacto da IA
    dados_mapa = pd.DataFrame({
        'Setor': ['Tecnologia', 'Saúde', 'Indústria', 'Educação', 'Agricultura'],
        'Região': ['América do Norte', 'Europa', 'Ásia', 'América Latina', 'África'],
        'Impacto': [80, 65, 70, 60, 50],
        'Latitude': [37.7749, 48.8566, 35.6895, -23.5505, -1.2921],
        'Longitude': [-122.4194, 2.3522, 139.6917, -46.6333, 36.8219]
    })

    st.write("**Dados de impacto geolocalizados:**")
    st.dataframe(dados_mapa)

    # Configuração do pydeck para criar bolhas proporcionais ao impacto
    layer = pdk.Layer(
        "ScatterplotLayer",
        dados_mapa,
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=5000,
        radius_min_pixels=5,
        radius_max_pixels=50,
        line_width_min_pixels=1,
        get_position='[Longitude, Latitude]',
        get_radius='Impacto',
        get_fill_color='[255 - Impacto, Impacto, 100]',
        get_line_color=[0, 0, 0]
    )

    # View inicial do mapa — AJUSTE DE ZOOM
    view_state = pdk.ViewState(
        latitude=20,      # Ajustado para focar mais acima do Equador
        longitude=0,      # Foco central do mundo
        zoom=2,           # Mais próximo do globo
        pitch=0
    )

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Setor: {Setor}\nRegião: {Região}\nImpacto: {Impacto}%"}
    )

    st.pydeck_chart(r)

    st.write("""
    Este **mapa mundi** interativo mostra, com o tamanho das bolhas e cores, o **nível de impacto da IA** em diferentes setores e regiões do mundo.
    """)
# Rodapé
st.sidebar.title("Sobre o Projeto")
st.sidebar.write("Explore o impacto da IA no mercado de trabalho.")
