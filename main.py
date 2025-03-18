import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Análise de Fraude em Cartões", layout="wide")

# Carregar o dataset
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")  # Substitua pelo caminho do seu dataset
    return df

df = load_data()

# Título do aplicativo
st.title("🔍 Análise de Fraude em Cartões de Crédito")

# Mostrar dataset
st.subheader("Visualizando os primeiros registros do dataset")
st.dataframe(df.head())

# Criar gráfico de barras mostrando o número de transações legítimas vs. fraudulentas
st.subheader("Distribuição de Transações")
fraud_count = df["Class"].value_counts()
st.bar_chart(fraud_count)

# Criar um histograma da distribuição do valor das transações
st.subheader("Distribuição do Valor das Transações")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["Amount"], bins=50, kde=True, ax=ax)
st.pyplot(fig)

# Criar um filtro para exibir apenas transações fraudulentas
st.subheader("Filtrar Transações Fraudulentas")
if st.checkbox("Mostrar apenas fraudes"):
    st.dataframe(df[df["Class"] == 1])

# Criar um gráfico da média do valor das transações ao longo do tempo
st.subheader("Média do Valor das Transações ao Longo do Dia")
df["Hour"] = (df["Time"] // 3600) % 24  # Converter tempo para horas
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df, x="Hour", y="Amount", hue="Class", estimator="mean", ax=ax)
st.pyplot(fig)

st.write("🚀 *Dashboard criado com Streamlit!*")