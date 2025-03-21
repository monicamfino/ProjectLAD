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
    df = df.dropna()
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Rolling_Mean_Amount"] = df["Amount"].rolling(window=5).mean()
    df["Std_Amount"] = df["Amount"].rolling(window=5).std()
    df["Delta_Amount"] = df["Amount"].diff()
    df["Amount_Category"] = pd.cut(df["Amount"], bins=[0, 10, 50, 100, 500, 5000, np.inf],
                                   labels=["Very Low", "Low", "Medium", "High", "Very High", "Extreme"])
    np.random.seed(42)
    df["Region"] = np.random.choice(["Norte", "Sul", "Leste", "Oeste", "Centro"], size=len(df))
    return df


df = load_data()

# Criar Sidebar (Menu lateral)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1041/1041916.png", width=100)
st.sidebar.title("🐰 BugsBunny Analytics")
st.sidebar.write("Soluções Inteligentes para a Detecção de Fraudes")
page = st.sidebar.radio("Navegação", [
    "🏠 Visão Geral",
    "📊 Análise de Fraudes",
    "📈 Estatísticas",
    "📂 Relatórios e Configurações"
])

# 📌 Página Inicial - Visão Geral
if page == "🏠 Visão Geral":
    st.markdown('<p class="big-font">🔍 Visão Geral - Detecção de Fraude</p>', unsafe_allow_html=True)

    # 🏢 Sobre a Plataforma
    st.subheader("💼 Sobre o BugsBunny Analytics")
    st.write("""
    A nossa missão é ajudar empresas a detectarem fraudes financeiras com inteligência artificial e análise de dados.
    Oferecemos soluções para monitorização, prevenção e identificação de atividades suspeitas.
    """)

    # 📜 Tipos Comuns de Fraude
    st.subheader("📜 Tipos Comuns de Fraude")
    fraud_types = pd.DataFrame({
        "Tipo de Fraude": ["Fraude em Cartão de Crédito", "Phishing", "Roubo de Identidade", "Transações Falsificadas"],
        "Descrição": [
            "Utilização não autorizada do cartão para compras.",
            "Enganar utilizadores para fornecerem informações sensíveis.",
            "Falsificação de identidade para acesso financeiro ilícito.",
            "Manipulação ou falsificação de transações bancárias."
        ]
    })
    st.table(fraud_types)

    # 📊 Estatísticas Gerais
    total_transacoes = len(df)
    transacoes_fraudulentas = df["Class"].sum()
    taxa_fraude = (transacoes_fraudulentas / total_transacoes) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("💳 Total de Transações", f"{total_transacoes:,}")
    col2.metric("⚠ Transações Fraudulentas", f"{transacoes_fraudulentas:,}")
    col3.metric("📉 Taxa de Fraude", f"{taxa_fraude:.2f} %")

    # 🛡️ Como Prevenir Fraudes?
    st.subheader("🛡️ Como Prevenir Fraudes?")
    st.write("""
    A prevenção de fraudes envolve um conjunto de boas práticas e tecnologias que ajudam a proteger empresas e consumidores. 
    Aqui estão algumas recomendações essenciais:
    """)

    fraud_prevention = pd.DataFrame({
        "Tipo de Fraude": ["Fraude em Cartão de Crédito", "Phishing", "Roubo de Identidade", "Transações Falsificadas"],
        "Medidas Preventivas": [
            "Ativar alertas de transação, usar autenticação multifator e monitoramento contínuo.",
            "Nunca compartilhar dados pessoais, verificar remetentes suspeitos e utilizar autenticação em dois fatores.",
            "Utilizar verificação biométrica, não reutilizar senhas e ativar bloqueios automáticos.",
            "Implementar monitoramento de transações em tempo real e análises de comportamento."
        ]
    })
    st.table(fraud_prevention)

    # 💡 Tecnologias e Estratégias para Prevenção
    st.subheader("💡 Tecnologias e Estratégias para Prevenção de Fraudes")
    st.write("""
    As empresas podem adotar as seguintes tecnologias para reforçar a segurança:
    - **Machine Learning & IA**: Modelos que analisam padrões e detectam anomalias.
    - **Autenticação Multifator (MFA)**: Verificação em duas etapas para acessos financeiros.
    - **Monitoramento em Tempo Real**: Identificação de transações suspeitas à medida que ocorrem.
    - **Criptografia Avançada**: Proteção de dados sensíveis contra acessos não autorizados.
    - **Análises de Comportamento**: Identificação de padrões incomuns de uso do sistema.
    """)

# Página 2: Análise de Fraudes
elif page == "📊 Análise de Fraudes":
    st.markdown('<p class="big-font">📊 Análise de Fraudes</p>', unsafe_allow_html=True)
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    # 🔥 Filtros Interativos
    st.subheader("🎯 Filtros de Análise")
    hora_selecionada = st.slider("Selecione um intervalo de horário", 0, 23, (0, 23))
    regiao_selecionada = st.multiselect("Filtrar por região", df["Region"].unique(), default=df["Region"].unique())

    fraude_filtrada = fraud[
        (fraud["Hour"].between(hora_selecionada[0], hora_selecionada[1])) &
        (fraud["Region"].isin(regiao_selecionada))
    ]

    # 📊 Gráfico: Fraudes ao Longo do Dia
    st.subheader("📆 Distribuição de Fraudes por Horário")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(fraude_filtrada["Hour"], bins=24, kde=True, color="red", ax=ax)
    ax.set_xlabel("Hora do Dia")
    ax.set_ylabel("Número de Fraudes")
    st.pyplot(fig)

    # 📍 Fraudes por Região
    st.subheader("🌍 Fraudes por Região")
    fraude_por_regiao = fraude_filtrada["Region"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=fraude_por_regiao.index, y=fraude_por_regiao.values, palette="Reds_r", ax=ax)
    ax.set_ylabel("Percentagem de Fraudes (%)")
    st.pyplot(fig)

    # 📈 Boxplot: Distribuição dos Valores Fraudulentos
    st.subheader("💰 Análise dos Valores das Fraudes")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=fraude_filtrada["Amount"], color="red", ax=ax)
    ax.set_xlabel("Valor da Fraude ($)")
    st.pyplot(fig)

    # 📊 Heatmap: Fraudes por Hora e Região
    st.subheader("🔥 Mapa de Calor: Fraudes por Hora e Região")
    heatmap_data = fraud.pivot_table(index="Region", columns="Hour", values="Class", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="Reds", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # 📌 Insights Automáticos
    st.subheader("📌 Insights Automáticos")
    if len(fraude_filtrada) > 0:
        max_hora = fraude_filtrada["Hour"].value_counts().idxmax()
        max_regiao = fraude_filtrada["Region"].mode()[0]
        st.write(f"📌 **A maior concentração de fraudes ocorre às {max_hora}h.**")
        st.write(f"📌 **A região mais afetada é {max_regiao}.**")
        st.write(f"📌 **O valor médio das fraudes é ${fraude_filtrada['Amount'].mean():.2f}.**")
        st.write(f"📌 **O maior valor de fraude registrado foi ${fraude_filtrada['Amount'].max():.2f}.**")
    else:
        st.write("✅ Nenhuma fraude encontrada para os filtros selecionados.")

    # 📤 Exportação de Dados
    st.subheader("📥 Exportar Dados Filtrados")
    csv_filtros = fraude_filtrada.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Baixar CSV", data=csv_filtros, file_name="fraudes_filtradas.csv", mime="text/csv")


# 📈 Página de Estatísticas
elif page == "📈 Estatísticas":
    st.markdown('<p class="big-font">📈 Estatísticas Avançadas</p>', unsafe_allow_html=True)

    st.subheader("📊 Médias e Medianas")
    col1, col2 = st.columns(2)
    col1.write("### Média:")
    col1.write(df.mean(numeric_only=True))
    col2.write("### Mediana:")
    col2.write(df.median(numeric_only=True))

    st.subheader("📊 Variância e Desvio Padrão")
    col1, col2 = st.columns(2)
    col1.write("### Variância:")
    col1.write(df.var(numeric_only=True))
    col2.write("### Desvio Padrão:")
    col2.write(df.std(numeric_only=True))

    # 🔥 Matriz de Correlação
    st.subheader("🔥 Matriz de Correlação")
    fig, ax = plt.subplots(figsize=(10, 8))
    df_numeric = df.select_dtypes(include=["number"])
    sns.heatmap(df_numeric.corr(), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    # 📊 Matriz de Covariância
    st.subheader("📊 Matriz de Covariância")
    st.write(df_numeric.cov())

    # 📌 Análise de Fraudes por Valor e Região
    st.subheader("💰 Fraudes por Valor e Região")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="Region", y="Amount", hue="Class", split=True, ax=ax)
    ax.set_xlabel("Região")
    ax.set_ylabel("Valor da Transação")
    st.pyplot(fig)

    # 📌 Insights Automáticos
    st.markdown("### 📌 Insights Automáticos")
    if df["Class"].sum() > 0:
        regiao_mais_fraudulenta = df[df["Class"] == 1]["Region"].mode()[0]
        valor_medio_fraude = df[df["Class"] == 1]["Amount"].mean()
        st.write(f"📌 **A região com mais fraudes é {regiao_mais_fraudulenta}.**")
        st.write(f"📌 **O valor médio de uma transação fraudulenta é de R$ {valor_medio_fraude:.2f}.**")
    else:
        st.write("Nenhuma fraude detectada nos dados.")

# Página 4: Relatórios e Configurações
elif page == "📂 Relatórios e Configurações":
    st.markdown('<p class="big-font">📂 Relatórios e Configurações</p>', unsafe_allow_html=True)

    # Definindo sub-páginas
    sub_page = st.sidebar.radio("Subtópicos", ["📑 Gerar Relatório", "⚙ Configurações Avançadas"])

    # 📑 Geração de Relatórios Personalizados
    if sub_page == "📑 Gerar Relatório":
        st.subheader("📥 Exportação de Dados")

        # 🎯 Filtros Avançados para o Relatório
        colunas_disponiveis = list(df.columns)
        colunas_selecionadas = st.multiselect("Selecione as colunas para o relatório", colunas_disponiveis,
                                              default=colunas_disponiveis)

        tipo_transacao = st.radio("Filtrar transações:", ["Todas", "Apenas Fraudes", "Apenas Legítimas"])

        if tipo_transacao == "Apenas Fraudes":
            df_export = df[df["Class"] == 1]
        elif tipo_transacao == "Apenas Legítimas":
            df_export = df[df["Class"] == 0]
        else:
            df_export = df.copy()

        df_export = df_export[colunas_selecionadas]

        # 📊 Visualizar os dados antes do download
        st.write("🔍 **Pré-visualização dos Dados:**")
        st.dataframe(df_export.head(10))

        # 📂 Opções de Exportação
        formato = st.selectbox("Escolha o formato do relatório:", ["CSV", "Excel"])
        if formato == "CSV":
            file_data = df_export.to_csv(index=False).encode('utf-8')
            file_name = "relatorio_fraude.csv"
            mime_type = "text/csv"
        else:
            file_data = df_export.to_excel(index=False).encode('utf-8')
            file_name = "relatorio_fraude.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        st.download_button(label=f"📥 Baixar {formato}", data=file_data, file_name=file_name, mime=mime_type)

    # ⚙ Configurações Avançadas
    elif sub_page == "⚙ Configurações Avançadas":
        st.subheader("⚙ Ajustes do Sistema")

        # 📌 Configuração de Alertas de Fraude
        limite_alerta = st.slider("Definir limite de alerta para transações suspeitas ($):", 10, 5000, 1000)
        metodo_analise = st.radio("Escolha o método de detecção de fraudes:", ["Regra Fixa", "Machine Learning"])

        # 🌍 Configuração de Regiões
        st.subheader("🌎 Personalizar Análise por Região")
        selected_region = st.multiselect("Selecione as regiões a monitorar:", df["Region"].unique(),
                                         default=df["Region"].unique())

        # 🎯 Aplicar configurações (Simulação)
        if st.button("Salvar Configurações"):
            st.success("✅ Configurações salvas com sucesso!")
            st.write(f"- **Limite de Alerta:** ${limite_alerta}")
            st.write(f"- **Método de Detecção:** {metodo_analise}")
            st.write(f"- **Regiões Monitoradas:** {', '.join(selected_region)}")



