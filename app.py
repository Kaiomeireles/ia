import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Impacto da IA no Mercado de Trabalho",
    page_icon="🤖",
    layout="wide"
)

# Menu lateral
menu = st.sidebar.selectbox(
    "Navegação",
    ["Introdução", "Base de Dados", "Análise Descritiva", "Inferência Estatística", "IA Integrada", "Mapa Geoespacial"]
)

# Função para carregar os dados reais do CSV
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

# Introdução
if menu == "Introdução":
    st.title("Introdução 🚀")
    st.write("""
    A inteligência artificial (IA) está transformando o mercado de trabalho global, criando uma nova dinâmica entre profissões emergentes e aquelas em declínio.
    Este projeto explora como a adoção da IA afeta diferentes setores e regiões, propondo análises baseadas em dados para compreender suas implicações.
    """)

    st.subheader("Problema e Contexto de Mercado")
    st.write("""
    A automação e a IA estão substituindo processos repetitivos, enquanto abrem espaço para inovações. No entanto, há preocupação com o deslocamento de empregos, desigualdade e a adaptação da força de trabalho global.
    """)

    st.subheader("Descrição da Base de Dados e Variáveis")
    st.write("""
    Os dados utilizados neste projeto incluem métricas sobre o impacto da IA em setores específicos, abrangendo:
    - **Setores**: Tecnologia, Saúde, Indústria, Educação, Agricultura.
    - **Impacto (%)**: Percentual estimado de automação em cada setor.
    - **Geolocalização**: Regiões mais afetadas.
    """)

    st.subheader("Perguntas e Hipóteses")
    st.write("""
    Este projeto visa responder às seguintes perguntas:
    - Quais setores são mais impactados pela adoção de IA?
    - Há correlação entre impacto e localização geográfica?
    - Como as tendências de automação afetam o mercado de trabalho a longo prazo?

    **Hipóteses:**
    - O setor de tecnologia é o mais afetado positivamente pela IA.
    - Regiões mais desenvolvidas têm maior adoção da automação.
    - Profissões criativas têm menor impacto da automação.
    """)

# Base de Dados + Intervalo de Confiança
elif menu == "Base de Dados":
    st.title("Base de Dados 📊")
    if not df.empty:
        setores = df['Setor'].unique()
        setor_escolhido = st.selectbox("Selecione o setor para calcular o intervalo de confiança:", setores)
        df_filtrado = df[df['Setor'] == setor_escolhido]

        st.subheader("🔍 Visualização dos Dados por Setor")
        st.dataframe(df_filtrado)
        st.markdown(f"**Setores únicos:** {', '.join(df['Setor'].unique())}")
        st.markdown(f"**Regiões únicas:** {', '.join(df['Região'].unique())}")

        dados_setor = df_filtrado['Impacto']

        media = np.mean(dados_setor)
        desvio = np.std(dados_setor, ddof=1)
        n = len(dados_setor)
        t_critico = stats.t.ppf(0.975, df=n-1)
        margem_erro = t_critico * (desvio / np.sqrt(n))
        ic_min = media - margem_erro
        ic_max = media + margem_erro

        st.subheader("📌 Interpretação Estatística")
        st.write("Estamos interessados na **média de impacto da IA** em cada setor. Como a variável 'Impacto' é numérica contínua, usamos o **intervalo de confiança para a média** com distribuição t de Student.")

        st.metric("Média de Impacto", f"{media:.2f}%")
        st.write(f"Intervalo de Confiança (95%): **[{ic_min:.2f}%, {ic_max:.2f}%]**")

        fig, ax = plt.subplots()
        ax.bar(setor_escolhido, media, yerr=margem_erro, capsize=10, color='#1f77b4')
        ax.set_ylabel('% de Impacto Estimado')
        ax.set_title(f'Intervalo de Confiança (95%) - {setor_escolhido}')
        ax.set_ylim(0, 100)
        st.pyplot(fig)

        st.success(f"Com 95% de confiança, a média de impacto da IA no setor **{setor_escolhido}** está entre {ic_min:.2f}% e {ic_max:.2f}%.")
    else:
        st.warning("Os dados não foram carregados corretamente. Verifique o arquivo 'dados.csv'.")

# Análise Descritiva
elif menu == "Análise Descritiva":
    st.title("Análise Descritiva 📈")
    st.write("Nesta seção, exploramos as estatísticas básicas e as visualizações de impacto por setor.")
    st.subheader("Estatísticas Básicas")
    st.write(df.describe())
    st.subheader("Gráfico de Impacto por Setor")
    st.bar_chart(df.set_index("Setor")["Impacto"])

# Inferência Estatística
elif menu == "Inferência Estatística":
    st.title("Inferência Estatística 📉")
    st.write("""
    Nesta seção, fazemos análises baseadas em hipóteses e calculamos intervalos de confiança.
    """)
    media = df["Impacto"].mean()
    st.write(f"Média de Impacto: {media:.2f}%")
    st.write("""
    **Exemplo de Hipótese:**
    - O setor de tecnologia apresenta impacto acima da média comparado aos demais setores.
    """)

# IA Integrada
elif menu == "IA Integrada":
    st.title("IA Integrada 🤖")
    st.write("""
    Aqui você pode interagir com uma IA para responder perguntas relacionadas ao impacto da automação no mercado de trabalho.
    """)
    pergunta = st.text_input("Digite sua pergunta:")
    if pergunta:
        st.write("Resposta da IA: **Funcionalidade em Desenvolvimento**")

# Mapa Geoespacial
elif menu == "Mapa Geoespacial":
    st.title("Mapa Geoespacial 🌍")
    st.write("""
    Visualize as regiões mais afetadas pela IA:
    - América do Norte: Indústria
    - Europa: Educação
    - América Latina: Agricultura
    """)
    st.write("**Nota:** Funcionalidade de mapas interativos será incluída futuramente.")

# Rodapé e Referências
st.header("Referências 📚")
st.markdown("""
- **World Bank Open Data**: Dados econômicos e sociais globais. [Link](https://data.worldbank.org/)
- **OECD Data**: Informações sobre mercado de trabalho e automação. [Link](https://www.oecd.org/)
- **Kaggle Datasets**: Conjuntos de dados variados sobre IA e empregos. [Link](https://www.kaggle.com/)
- **McKinsey Global Institute**: Relatórios técnicos sobre IA e automação. [Link](https://www.mckinsey.com/featured-insights/future-of-work)
""")

st.sidebar.title("Sobre o Projeto")
st.sidebar.write("""
Navegue pelo menu para explorar o impacto da IA em diferentes setores e localidades.
""")