import streamlit as st
import pandas as pd

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

# Introdução
if menu == "Introdução":
    st.title("Introdução 🚀")
    st.write("""
    A inteligência artificial (IA) está transformando o mercado de trabalho global, criando uma nova dinâmica entre profissões emergentes e aquelas em declínio.
    Este projeto explora como a adoção da IA afeta diferentes setores e regiões, propondo análises baseadas em dados para compreender suas implicações.
    """)

    # Contexto do Problema e Mercado
    st.subheader("Problema e Contexto de Mercado")
    st.write("""
    A automação e a IA estão substituindo processos repetitivos, enquanto abrem espaço para inovações. No entanto, há preocupação com o deslocamento de empregos, desigualdade e a adaptação da força de trabalho global.
    """)
    
    # Descrição da Base de Dados
    st.subheader("Descrição da Base de Dados e Variáveis")
    st.write("""
    Os dados utilizados neste projeto incluem métricas sobre o impacto da IA em setores específicos, abrangendo:
    - **Setores**: Tecnologia, Saúde, Indústria, Educação, Agricultura.
    - **Impacto (%)**: Percentual estimado de automação em cada setor.
    - **Geolocalização**: Regiões mais afetadas.
    """)

    # Identificação de Perguntas e Hipóteses
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

# Base de Dados
elif menu == "Base de Dados":
    st.title("Base de Dados 📊")
    st.write("Aqui está um exemplo de base de dados utilizada nas análises:")
    data = {
        "Setor": ["Tecnologia", "Saúde", "Indústria", "Educação", "Agricultura"],
        "Impacto (%)": [70, 50, 40, 35, 20],
        "Região": ["Global", "Global", "América do Norte", "Europa", "América Latina"]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)
    st.write("""
    **Descrição das Variáveis:**
    - **Setor**: Indústria analisada.
    - **Impacto (%)**: Percentual estimado de automação e integração da IA.
    - **Região**: Localidades afetadas pelo impacto analisado.
    """)

# Análise Descritiva
elif menu == "Análise Descritiva":
    st.title("Análise Descritiva 📈")
    st.write("Nesta seção, exploramos as estatísticas básicas e as visualizações de impacto por setor.")
    st.subheader("Estatísticas Básicas")
    st.write(df.describe())
    st.subheader("Gráfico de Impacto por Setor")
    st.bar_chart(df.set_index("Setor")["Impacto (%)"])

# Inferência Estatística
elif menu == "Inferência Estatística":
    st.title("Inferência Estatística 📉")
    st.write("""
    Nesta seção, fazemos análises baseadas em hipóteses e calculamos intervalos de confiança.
    """)
    media = df["Impacto (%)"].mean()
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

st.header("Referências 📚")
st.markdown("""
- **World Bank Open Data**: Dados econômicos e sociais globais. [Link](https://data.worldbank.org/)
- **OECD Data**: Informações sobre mercado de trabalho e automação. [Link](https://www.oecd.org/)
- **Kaggle Datasets**: Conjuntos de dados variados sobre IA e empregos. [Link](https://www.kaggle.com/)
- **McKinsey Global Institute**: Relatórios técnicos sobre IA e automação. [Link](https://www.mckinsey.com/featured-insights/future-of-work)
""")


# Rodapé
st.sidebar.title("Sobre o Projeto")
st.sidebar.write("""
Navegue pelo menu para explorar o impacto da IA em diferentes setores e localidades.
""")
