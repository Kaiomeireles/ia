import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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

# Introdução
if menu == "Introdução":
    st.title("Introdução 🚀")
    st.write("""
    A inteligência artificial (IA) está transformando o mercado de trabalho global, criando uma nova dinâmica entre profissões emergentes e aquelas em declínio.
    Este projeto explora como a adoção da IA afeta diferentes setores e regiões, propondo análises baseadas em dados para compreender suas implicações.
    """)

# Base de Dados + Intervalo de Confiança
elif menu == "Base de Dados":
    st.title("Base de Dados 📊")
    if not df.empty:
        st.dataframe(df)
        setor_escolhido = st.selectbox("Selecione o setor:", df['Setor'].unique())
        df_filtrado = df[df['Setor'] == setor_escolhido]

        # Estatísticas
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
    
    # Como não tem Latitude, vamos criar uma variável numérica a partir da Região
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
    st.title("IA Integrada 🤖")
    st.write("Interaja com a IA sobre impacto da automação.")
    pergunta = st.text_input("Faça sua pergunta:")
    if pergunta:
        st.info("Resposta automática: A IA impacta de forma diferente conforme o setor e a região.")

# Mapa Geoespacial
elif menu == "Mapa Geoespacial":
    st.title("Mapa Geoespacial 🌍")
    st.write("Sem dados de geolocalização disponíveis para gerar o mapa.")
    st.warning("Para visualizar o mapa, adicione as colunas 'Latitude' e 'Longitude' ao CSV.")

# Rodapé
st.sidebar.title("Sobre o Projeto")
st.sidebar.write("Explore o impacto da IA no mercado de trabalho.")
