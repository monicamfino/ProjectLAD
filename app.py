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

    # Criar novas features
    df["Rolling_Mean_Amount"] = df["Amount"].rolling(window=5).mean()
    df["Std_Amount"] = df["Amount"].rolling(window=5).std()
    df["Delta_Amount"] = df["Amount"].diff()
    df["Amount_Category"] = pd.cut(df["Amount"], bins=[0, 10, 50, 100, 500, 5000, np.inf],
                                   labels=["Very Low", "Low", "Medium", "High", "Very High", "Extreme"])
    df["Time_Diff"] = df["Time"].diff()
    df["Transacao_Noturna"] = df["Hour"].apply(lambda x: 1 if x < 6 else 0)
    df["Num_Transacoes_1h"] = df.groupby("Hour")["Time"].transform("count")
    df["Freq_Valor_Transacao"] = df.groupby("Amount")["Amount"].transform("count")
    df["Delta_Media_Valor"] = df["Amount"] - df["Rolling_Mean_Amount"]

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
    "📂 Relatórios e Configurações",
    "🧭 Navegação"
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

    # 🛠️ Variáveis Utilizadas no Modelo e no CSV
    st.subheader("🛠️ Variáveis Utilizadas no Modelo e no CSV")
    variaveis_combinadas = pd.DataFrame({
        "Variável": [
            "Time", "V1-V28", "Amount", "Class",
            "Hour", "Rolling_Mean_Amount", "Std_Amount", "Delta_Amount",
            "Amount_Category", "Time_Diff", "Transacao_Noturna",
            "Num_Transacoes_1h", "Freq_Valor_Transacao", "Delta_Media_Valor", "Region"
        ],
        "Descrição": [
            "Tempo decorrido desde a primeira transação no dataset.",
            "Variáveis anonimizadas resultantes de PCA (28 componentes principais).",
            "Montante da transação.",
            "Classe da transação (0: Legítima, 1: Fraudulenta).",
            "Hora do dia em que a transação ocorreu.",
            "Média móvel do valor da transação (janela de 5 transações).",
            "Desvio padrão do valor da transação (janela de 5 transações).",
            "Diferença entre o valor atual e o valor anterior da transação.",
            "Categoria do valor da transação (ex.: Muito Baixo, Baixo, Médio, etc.).",
            "Diferença de tempo entre transações consecutivas.",
            "Indica se a transação ocorreu durante a noite (1: Sim, 0: Não).",
            "Número de transações realizadas na mesma hora.",
            "Frequência de transações com o mesmo valor.",
            "Diferença entre o valor da transação e a média móvel.",
            "Região geográfica associada à transação."
        ]
    })
    st.table(variaveis_combinadas)

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

    # Explicação sobre as correlações
    st.write("""
    📌 **Análise das Correlações:**
    - **Correlações Positivas Fortes:**
    - Rolling_Mean_Amount e Std_Amount: Correlação positiva forte
    - Num_Transacoes_1h e algumas variáveis V: Correlações positivas moderadas

    - **Correlações Negativas Importantes:**
    - Delta_Amount e Rolling_Mean_Amount: Correlação negativa moderada
    - Time_Diff e algumas variáveis V: Correlações negativas moderadas
    - Delta_Media_Valor e Amount: Diferenças em relação à média tendem a ser inversas ao valor total da transação
    """)

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

        # 📊 Visualizar os dados antes do download
        st.write("🔍 **Pré-visualização dos Dados:**")
        st.dataframe(df_export.head(10))

        # 📊 Distribuição de Categorias de Montante
        st.subheader("📊 Distribuição de Categorias de Montante")
        fig, ax = plt.subplots(figsize=(8, 4))
        df["Amount_Category"].value_counts().plot(kind="bar", color="skyblue", ax=ax)
        ax.set_xlabel("Categoria de Montante")
        ax.set_ylabel("Número de Transações")
        st.pyplot(fig)

        # 🌙 Proporção de Transações Noturnas
        st.subheader("🌙 Proporção de Transações Noturnas")
        transacao_noturna = df["Transacao_Noturna"].value_counts(normalize=True) * 100
        st.write(f"**Transações Noturnas:** {transacao_noturna[1]:.2f}%")
        st.write(f"**Transações Diurnas:** {transacao_noturna[0]:.2f}%")

        # 📈 Média Móvel do Montante
        st.subheader("📈 Média Móvel do Montante")
        fig, ax = plt.subplots(figsize=(10, 5))
        df["Rolling_Mean_Amount"].plot(ax=ax, color="blue", label="Média Móvel (5 Transações)")
        ax.set_xlabel("Índice")
        ax.set_ylabel("Montante ($)")
        ax.legend()
        st.pyplot(fig)

        # ⏱️ Diferença de Tempo entre Transações
        st.subheader("⏱️ Diferença de Tempo entre Transações")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["Time_Diff"].dropna(), bins=30, kde=True, color="purple", ax=ax)
        ax.set_xlabel("Diferença de Tempo (segundos)")
        ax.set_ylabel("Frequência")
        st.pyplot(fig)

        # 🔥 Mapa de Calor: Número de Transações por Hora e Região
        st.subheader("🔥 Mapa de Calor: Número de Transações por Hora e Região")
        heatmap_data = df.pivot_table(index="Region", columns="Hour", values="Num_Transacoes_1h", aggfunc="mean",
                                      fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap="Blues", linewidths=0.5, ax=ax)
        st.pyplot(fig)

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

# Nova página: Navegação
if page == "🧭 Navegação":
    st.markdown('<p class="big-font">🧭 Navegação</p>', unsafe_allow_html=True)

    st.subheader("📊 Dashboard de Variáveis")

    # Exibir as variáveis e seus valores
    variaveis_valores = {
        "Total de Transações": len(df),
        "Transações Fraudulentas": df["Class"].sum(),
        "Taxa de Fraude (%)": (df["Class"].sum() / len(df)) * 100,
        "Valor Médio das Transações ($)": df["Amount"].mean(),
        "Valor Máximo das Transações ($)": df["Amount"].max(),
        "Valor Mínimo das Transações ($)": df["Amount"].min(),
        "Desvio Padrão do Valor das Transações ($)": df["Amount"].std()
    }
    
    for variavel, valor in variaveis_valores.items():
        st.metric(label=variavel, value=f"{valor:,.2f}" if isinstance(valor, float) else f"{valor:,}")
