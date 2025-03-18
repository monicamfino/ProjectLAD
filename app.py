import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Configuração da página
st.set_page_config(page_title="BugsBunny - Detecção de Fraude 💳", layout="wide")

# Estilizando o app
st.markdown("""
    <style>
        .big-font { font-size:30px !important; font-weight: bold; }
        .stApp { background-color: #f5f5f5; }
    </style>
""", unsafe_allow_html=True)

# Carregar os dados
@st.cache_data
def load_data():
    time.sleep(2)
    df = pd.read_csv("creditcard.csv")

    # Tratamento de dados
    df = df.dropna()
    df = df.fillna(df.mean())
    df["Hour"] = (df["Time"] // 3600) % 24  # Convertendo o tempo para horas do dia

    # Criando novas features
    df["Rolling_Mean_Amount"] = df["Amount"].rolling(window=5).mean()
    df["Std_Amount"] = df["Amount"].rolling(window=5).std()
    df["Delta_Amount"] = df["Amount"].diff()
    df["Amount_Category"] = pd.cut(df["Amount"], bins=[0, 10, 50, 100, 500, 5000, np.inf],
                                   labels=["Very Low", "Low", "Medium", "High", "Very High", "Extreme"])

    return df

df = load_data()

# Criar Sidebar (Menu lateral)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1041/1041916.png", width=100)
st.sidebar.title("🐰 BugsBunny Analytics")
st.sidebar.write("Soluções Inteligentes para Detecção de Fraudes")
page = st.sidebar.radio("Navegação", ["🏠 Visão Geral", "📊 Análise de Fraudes", "📈 Estatísticas"])

# Página 1: Visão Geral
if page == "🏠 Visão Geral":
    st.markdown('<p class="big-font">🔍 BugsBunny - Análise de Fraude em Cartões</p>', unsafe_allow_html=True)
    st.write("Dashboard interativo desenvolvido pela **BugsBunny Analytics**, líder em segurança financeira.")

    # KPIs
    total_transacoes = len(df)
    transacoes_fraudulentas = df["Class"].sum()
    taxa_fraude = (transacoes_fraudulentas / total_transacoes) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("💳 Total de Transações", f"{total_transacoes:,}")
    col2.metric("⚠️ Transações Fraudulentas", f"{transacoes_fraudulentas:,}")
    col3.metric("📉 Taxa de Fraude", f"{taxa_fraude:.2f} %")

    # Gráfico de distribuição de fraudes
    st.subheader("Distribuição de Transações (Fraude vs. Normal)")
    fraud_count = df["Class"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=fraud_count.index, y=fraud_count.values, palette=["blue", "red"], ax=ax)
    ax.set_xticklabels(["Legítima", "Fraude"])
    ax.set_ylabel("Quantidade")
    st.pyplot(fig)

# Página 2: Análise de Fraudes
elif page == "📊 Análise de Fraudes":
    st.markdown('<p class="big-font">📊 BugsBunny - Análise de Fraudes</p>', unsafe_allow_html=True)

    # Comparação da média do valor da transação
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    col1, col2 = st.columns(2)
    col1.metric("💰 Média do Valor - Fraudes", f"${fraud['Amount'].mean():,.2f}")
    col2.metric("💳 Média do Valor - Legítimas", f"${legit['Amount'].mean():,.2f}")

    # Distribuição de valores de transação
    st.subheader("Distribuição dos Valores de Transação")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(fraud["Amount"], bins=50, color="red", label="Fraude", kde=True, ax=ax)
    sns.histplot(legit["Amount"], bins=50, color="blue", label="Legítima", kde=True, ax=ax)
    plt.legend()
    st.pyplot(fig)

    # Boxplot Comparativo
    st.subheader("Distribuição do Valor da Transação por Classe")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=df["Class"], y=df["Amount"], ax=ax)
    ax.set_xticklabels(["Legítima", "Fraude"])
    st.pyplot(fig)

    # Média do valor das transações ao longo do tempo
    st.subheader("Média do Valor das Transações ao Longo do Dia")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x="Hour", y="Amount", hue="Class", estimator="mean", ax=ax)
    st.pyplot(fig)

# Página 3: Estatísticas
elif page == "📈 Estatísticas":
    st.markdown('<p class="big-font">📈 Estatísticas Avançadas</p>', unsafe_allow_html=True)

    # Médias e Medianas
    st.subheader("📊 Médias e Medianas")
    col1, col2 = st.columns(2)
    col1.write("### Média:")
    col1.write(df.mean(numeric_only=True))

    col2.write("### Mediana:")
    col2.write(df.median(numeric_only=True))


    # Variância e Desvio Padrão
    st.subheader("📊 Variância e Desvio Padrão")
    col1, col2 = st.columns(2)
    col1.write("### Variância:")
    col1.write(df.var(numeric_only=True))
    col2.write("### Desvio Padrão:")
    col2.write(df.std(numeric_only=True))

    # Matriz de Correlação
    st.subheader("🔥 Matriz de Correlação")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    # Matriz de Covariância
    st.subheader("📊 Matriz de Covariância")
    st.write(df.cov())

st.pyplot(fig)

st.write("🚀 *Dashboard criado com Streamlit!*")
