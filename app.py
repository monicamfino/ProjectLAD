import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn import tree
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


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
st.sidebar.write("Deteção inteligente de fraudes para proteger o seu negócio.")
page = st.sidebar.radio("Navegação", [
    "🏠 Visão Geral",
    "📊 Análise de Fraudes",
    "📈 Estatísticas",
    "📂 Relatórios e Configurações",
    "🧭 Dados",
    "🤖 Machine Learning",
    "🧪 Classificar Transação"
])

# 📌 Página Inicial - Visão Geral
if page == "🏠 Visão Geral":
    st.markdown('<p class="big-font">🔍 Visão Geral - Como identificamos fraudes</p>', unsafe_allow_html=True)

    # 🏢 Sobre a Plataforma
    st.subheader("💼 Sobre o BugsBunny Analytics")
    st.write("""
    Ajudamos empresas a identificar transações suspeitas com ferramentas inteligentes e fáceis de usar.
Combinamos tecnologia e análise de dados para tornar a deteção de fraudes mais simples e eficaz.
    """)

    # 📜 Tipos Comuns de Fraude
    st.subheader("📜 Fraudes mais comuns que pode encontrar")
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
    st.subheader("🛠️ Informações analisadas em cada transação")
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
    st.subheader("🛡️  Como pode proteger-se contra fraudes?")
    st.write("""
    Evitar fraudes é possível com pequenos cuidados e algumas ferramentas tecnológicas. 
    Veja abaixo as recomendações que mais ajudam no dia a dia.
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
    st.subheader("💡 Como a tecnologia ajuda a prevenir fraudes")
    st.write("""
   Estas são algumas ferramentas usadas por empresas para manterem as suas transações seguras:
- **Inteligência Artificial**: Deteta padrões estranhos automaticamente.
- **Verificação em duas etapas**: Protege acessos com mais segurança.
- **Alertas em tempo real**: Detetam ações suspeitas à medida que acontecem.
- **Encriptação de dados**: Mantém as informações protegidas contra roubos.
- **Análise de comportamento**: Identifica mudanças no padrão de utilização.
    """)

# Página 2: Análise de Fraudes
elif page == "📊 Análise de Fraudes":
    st.markdown('<p class="big-font">📊 Análise de Fraudes</p>', unsafe_allow_html=True)
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    # 🔥 Filtros Interativos
    st.subheader("🎯 Ajustar a visualização")
    hora_selecionada = st.slider("Escolha o intervalo de horas a analisar", 0, 23, (0, 23))
    regiao_selecionada = st.multiselect("Escolha as regiões a incluir", df["Region"].unique(), default=df["Region"].unique())

    fraude_filtrada = fraud[
        (fraud["Hour"].between(hora_selecionada[0], hora_selecionada[1])) &
        (fraud["Region"].isin(regiao_selecionada))
        ]

    # 📊 Gráfico: Fraudes ao Longo do Dia
    st.subheader("📆 Em que horas ocorrem mais fraudes?")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(fraude_filtrada["Hour"], bins=24, kde=True, color="red", ax=ax)
    ax.set_xlabel("Hora do Dia")
    ax.set_ylabel("Número de Fraudes")
    st.pyplot(fig)

    # 📍 Fraudes por Região
    st.subheader("🌍 Onde ocorrem mais fraudes?")
    fraude_por_regiao = fraude_filtrada["Region"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=fraude_por_regiao.index, y=fraude_por_regiao.values, palette="Reds_r", ax=ax)
    ax.set_ylabel("Percentagem de Fraudes (%)")
    st.pyplot(fig)

    # 📈 Boxplot: Distribuição dos Valores Fraudulentos
    st.subheader("💰Quais os valores mais comuns nas fraudes?")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=fraude_filtrada["Amount"], color="red", ax=ax)
    ax.set_xlabel("Valor da Fraude ($)")
    st.pyplot(fig)

    # 📊 Heatmap: Fraudes por Hora e Região
    st.subheader("🔥  Quando e onde as fraudes mais acontecem?")
    heatmap_data = fraud.pivot_table(index="Region", columns="Hour", values="Class", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="Reds", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # 📌 Insights Automáticos
    st.subheader("📌 O que podemos observar?")
    if len(fraude_filtrada) > 0:
        max_hora = fraude_filtrada["Hour"].value_counts().idxmax()
        max_regiao = fraude_filtrada["Region"].mode()[0]
        st.write(f"📌 **A maior concentração de fraudes ocorre às {max_hora}h.**")
        st.write(f"📌 **A região mais afetada é {max_regiao}.**")
        st.write(f"📌 **O valor médio das fraudes é ${fraude_filtrada['Amount'].mean():.2f}.**")
        st.write(f"📌 **O maior valor de fraude registado foi ${fraude_filtrada['Amount'].max():.2f}.**")
    else:
        st.write("✅ Nenhuma fraude encontrada para os filtros selecionados.")

    # 📤 Exportação de Dados
    st.subheader("📥 Guardar resultados filtrados")
    csv_filtros = fraude_filtrada.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Descarregar ficheiro CSV", data=csv_filtros, file_name="fraudes_filtradas.csv", mime="text/csv")


# 📈 Página de Estatísticas
elif page == "📈 Estatísticas":
    st.markdown('<p class="big-font">📈 Estatísticas das Transações</p>', unsafe_allow_html=True)

    st.subheader("📊 Tendência Geral das Transações")
    col1, col2 = st.columns(2)
    col1.write("### Valor Médio:")
    col1.write(df.mean(numeric_only=True))
    col2.write("### Valor Mediano:")
    col2.write(df.median(numeric_only=True))

    st.subheader("📊 Variação dos Valores das Transações")
    col1, col2 = st.columns(2)
    col1.write("### Variação (Variância):")
    col1.write(df.var(numeric_only=True))
    col2.write("### Dispersão (Desvio Padrão):")
    col2.write(df.std(numeric_only=True))

    # 🔥 Matriz de Correlação
    st.subheader("🔥 Ligações entre Variáveis")
    fig, ax = plt.subplots(figsize=(10, 8))
    df_numeric = df.select_dtypes(include=["number"])
    sns.heatmap(df_numeric.corr(), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    # Explicação sobre as correlações
    st.write("""
    📌 **O que observamos nestas ligações:**
    - Algumas variáveis como a média e o desvio do valor tendem a andar juntas.
    - Certas variações no valor da transação podem indicar comportamentos menos comuns.
    """)

    # 📊 Matriz de Covariância
    st.subheader("📊 Matriz de Covariância")
    st.write(df_numeric.cov())

    # 📌 Análise de Fraudes por Valor e Região
    st.subheader("💰 Em que regiões ocorrem fraudes com maiores valores?")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="Region", y="Amount", hue="Class", split=True, ax=ax)
    ax.set_xlabel("Região")
    ax.set_ylabel("Valor da Transação")
    st.pyplot(fig)

    # 📌 Insights Automáticos
    st.markdown("### 📌 O que aprendemos com estes dados?")
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
    sub_page = st.sidebar.radio("Subtópicos", ["📑 Gerar Relatório", "⚙ Configurações Avançadas", "🧪 Comparação Visual de Variáveis"])

    # 📑 Geração de Relatórios Personalizados
    if sub_page == "📑 Gerar Relatório":
        st.subheader("📥 Gerar Relatório Personalizado")

        # 🎯 Filtros Avançados para o Relatório
        st.markdown("### 🎯 Escolha o que quer incluir no seu relatório")

        # 🎯 Filtros Avançados para o Relatório
        colunas_disponiveis = list(df.columns)
        colunas_selecionadas = st.multiselect("Selecione os dados que pretende visualizar:", colunas_disponiveis,
                                              default=colunas_disponiveis)

        tipo_transacao = st.radio("Tipo de transações a incluir:", ["Todas", "Apenas Fraudes", "Apenas Legítimas"])

        if tipo_transacao == "Apenas Fraudes":
            df_export = df[df["Class"] == 1]
        elif tipo_transacao == "Apenas Legítimas":
            df_export = df[df["Class"] == 0]
        else:
            df_export = df.copy()

        df_export = df_export[colunas_selecionadas]

        # 📊 Visualizar os dados antes do download
        st.write("🔍 **Pré-visualização das Transações Selecionadas:**")
        st.dataframe(df_export.head(10))

        # 📊 Distribuição de Categorias de Montante
        st.subheader("📊  Quantidade de Transações por Categoria de Valor")
        fig, ax = plt.subplots(figsize=(8, 4))
        df["Amount_Category"].value_counts().plot(kind="bar", color="skyblue", ax=ax)
        ax.set_xlabel("Categoria de Valor")
        ax.set_ylabel("Número de Transações")
        st.pyplot(fig)

        # 🌙 Proporção de Transações Noturnas
        st.subheader("🌙 Transações Noturnas e Diurnas")
        transacao_noturna = df["Transacao_Noturna"].value_counts(normalize=True) * 100
        st.write(f"**Transações Noturnas:** {transacao_noturna[1]:.2f}%")
        st.write(f"**Transações Diurnas:** {transacao_noturna[0]:.2f}%")

        # 📈 Média Móvel do Montante
        st.subheader("📈 Tendência da Média dos Valores")
        fig, ax = plt.subplots(figsize=(10, 5))
        df["Rolling_Mean_Amount"].plot(ax=ax, color="blue", label="Média Móvel (últimas 5 Transações)")
        ax.set_xlabel("Transações")
        ax.set_ylabel("Valor ($)")
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
        st.markdown("### 💾 Exportar Relatório")
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
        st.subheader("⚙ Preferências do Sistema")

        # 📌 Configuração de Alertas de Fraude
        limite_alerta = st.slider("Definir Alerta de Transações Suspeitas ($):", 10, 5000, 1000)
        metodo_analise = st.radio("Escolha o método de detecção de fraudes:", ["Regra Fixa", "Machine Learning"])

        # 🌍 Configuração de Regiões
        st.subheader("🌎 Personalizar Análise por Região")
        selected_region = st.multiselect("Selecione as regiões a monitorar:", df["Region"].unique(),
                                         default=df["Region"].unique())

        # 🎯 Aplicar configurações (Simulação)
        if st.button("Guardar Preferências"):
            st.success("✅ Configurações salvas com sucesso!")
            st.write(f"- **Limite de Alerta:** ${limite_alerta}")
            st.write(f"- **Tipo de verificação:** {metodo_analise}")
            st.write(f"- **Regiões monitorizadas:** {', '.join(selected_region)}")

    #  Normalização e Padronização
    elif sub_page == "🧪 Comparação Visual de Variáveis":
        st.subheader("🧪 Comparação Visual de Variáveis")

        st.write("""
        Antes de analisarmos os dados, é importante garantir que diferentes variáveis estejam numa escala comparável.
        🔍 Esta visualização permite observar como transações legítimas e fraudulentas se distribuem em função de duas variáveis escolhidas.

        💡 *Nota:* Algumas variáveis podem já ter sido normalizadas ou padronizadas para facilitar a comparação visual.
        """)

        st.write("Selecione duas variáveis para visualizar como as transações legítimas e fraudulentas se distribuem.")

        
        # Selecionar duas colunas para visualização
        col1, col2 = st.columns(2)
        with col1:
            selected_column1 = st.selectbox("Selecione a primeira coluna:", 
                                           df.select_dtypes(include=['number']).columns,
                                           key="vis_column1")
        with col2:
            selected_column2 = st.selectbox("Selecione a segunda coluna:", 
                                           df.select_dtypes(include=['number']).columns,
                                           key="vis_column2")
        
        # Amostrar dados para evitar sobrecarga
        sample_size = min(1000, len(df))
        sample_df = df.sample(sample_size, random_state=42)
        
        # Dados originais
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sample_df[selected_column1], sample_df[selected_column2], 
                   c=sample_df['Class'], cmap='coolwarm', alpha=0.6)
        ax.set_xlabel(selected_column1)
        ax.set_ylabel(selected_column2)
        ax.set_title("Antes da Transformação")
        # Adicionar legenda manual
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Fraude')
        blue_patch = mpatches.Patch(color='blue', label='Legítima')
        ax.legend(handles=[red_patch, blue_patch])
        
        st.pyplot(fig)
        
        # Dados padronizados
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        # Preparar os dados
        X = sample_df[[selected_column1, selected_column2]].values
        y = sample_df['Class'].values
        
        # Padronizar
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        
        # Normalizar
        min_max_scaler = MinMaxScaler()
        X_norm = min_max_scaler.fit_transform(X)
        
        # Plotar dados padronizados
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Padronizado
        ax[0].scatter(X_std[:, 0], X_std[:, 1], c=y, cmap='coolwarm', alpha=0.6)
        ax[0].set_xlabel(f"{selected_column1} (padronizado)")
        ax[0].set_ylabel(f"{selected_column2} (padronizado)")
        ax[0].set_title("Dados Padronizados")
        
        # Normalizado
        ax[1].scatter(X_norm[:, 0], X_norm[:, 1], c=y, cmap='coolwarm', alpha=0.6)
        ax[1].set_xlabel(f"{selected_column1} (normalizado)")
        ax[1].set_ylabel(f"{selected_column2} (normalizado)")
        ax[1].set_title("Dados Normalizados")
        
        plt.tight_layout()
        st.pyplot(fig)
        

# Nova página: Dados
elif page == "🧭 Dados":
    st.markdown('<p class="big-font">🧭 Dados</p>', unsafe_allow_html=True)

    st.subheader("📊 Visão Geral das Transações")

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

    # Adicionar scope das variáveis
    st.subheader("📄 Descrição das Variáveis")

    variaveis_escopo = {
        "Time": "Tempo decorrido desde a primeira transação no dataset.",
        "Vx": "Variáveis anonimizadas resultantes de PCA (28 componentes principais).",
        "Amount": "Montante da transação.",
        "Class": "Classe da transação (0: Legítima, 1: Fraudulenta).",
        "Hour": "Hora do dia em que a transação ocorreu.",
        "Rolling_Mean_Amount": "Média móvel do valor da transação (janela de 5 transações).",
        "Std_Amount": "Desvio padrão do valor da transação (janela de 5 transações).",
        "Delta_Amount": "Diferença entre o valor atual e o valor anterior da transação.",
        "Amount_Category": "Categoria do valor da transação (ex.: Muito Baixo, Baixo, Médio, etc.).",
        "Time_Diff": "Diferença de tempo entre transações consecutivas.",
        "Transacao_Noturna": "Indica se a transação ocorreu durante a noite (1: Sim, 0: Não).",
        "Num_Transacoes_1h": "Número de transações realizadas na mesma hora.",
        "Freq_Valor_Transacao": "Frequência de transações com o mesmo valor.",
        "Delta_Media_Valor": "Diferença entre o valor da transação e a média móvel.",
        "Region": "Região geográfica associada à transação."
    }

    for variavel, descricao in variaveis_escopo.items():
        st.write(f"**{variavel}:** {descricao}")

    # Adicionar gráficos de valores mínimo e máximo
    st.subheader("📊 Comparativo de Valores Mínimos e Máximos")

    # Agregar dados de V1-V28 em Vx
    df["Vx"] = df[[f"V{i}" for i in range(1, 29)]].sum(axis=1)
    min_vals = df[["Vx", "Hour", "Time_Diff"]].min()
    max_vals = df[["Vx", "Hour", "Time_Diff"]].max()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(min_vals.index, min_vals.values, color="blue", label="Min")
    ax.bar(max_vals.index, max_vals.values, color="red", label="Max", alpha=0.7)
    ax.set_title("Valores Mínimo e Máximo de Vx, Hour e Time_Diff")
    ax.set_ylabel("Valores")
    ax.legend()
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Adicionar gráficos de valores mínimo e máximo
    st.subheader("📊 Amplitude das Outras Variáveis Numéricas")

    # Remover colunas indesejadas
    columns_to_exclude = [f"V{i}" for i in range(1, 29)] + ["Vx", "Hour", "Time_Diff", "Class", "Transacao_Noturna"]
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_columns = [col for col in numeric_columns if col not in columns_to_exclude]

    # Calcular valores mínimos e máximos apenas para colunas numéricas filtradas
    min_vals = df[numeric_columns].min()
    max_vals = df[numeric_columns].max()

    # Garantir que os índices sejam strings
    min_vals.index = min_vals.index.astype(str)
    max_vals.index = max_vals.index.astype(str)

    # Garantir que os valores sejam numéricos
    min_vals = pd.to_numeric(min_vals, errors='coerce').fillna(0)
    max_vals = pd.to_numeric(max_vals, errors='coerce').fillna(0)

    # Criar o gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(min_vals.index, min_vals.values, color="blue", label="Min")
    ax.bar(max_vals.index, max_vals.values, color="red", label="Max", alpha=0.7)
    ax.set_title("Valores Mínimo e Máximo das Variáveis Numéricas")
    ax.set_ylabel("Valores")
    ax.legend()
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Adicionar legenda explicativa
    st.markdown("""
    **Legenda:**
    - **Min**: Valor mais baixo registado para a variável.
    - **Max**:  Valor mais alto registado para a variável.
    Estes indicadores ajudam a perceber a variação e o alcance dos dados analisados.
    """)

# Nova página: Machine Learning
elif page == "🤖 Machine Learning":
    # Adicionar tabs para diferentes modelos de ML
    model_tabs = st.tabs(["Introdução", "Classificação", "Ridge e Lasso Regression"])
    
    with model_tabs[0]:
        # Mover o conteúdo existente sobre ML para esta tab
        st.markdown("## Introdução ao Machine Learning")
        
        # Conceitos básicos
        st.subheader("🔍 O que é Machine Learning?")
        st.write("""
        O Machine Learning permite que os computadores **reconheçam padrões automaticamente** a partir de dados — sem precisarmos dizer exatamente o que fazer em cada situação.
        
        Em vez de programar regras fixas, o sistema **aprende com exemplos anteriores** para prever ou tomar decisões em situações futuras.
        """)
        
        # Comparação visual entre programação tradicional e ML
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💻 Programação Tradicional")
            st.markdown("""
            ```
            Dados + Regras → Resultados
            ```
            """)
            st.write("As regras são definidas pelo programador")
            
        with col2:
            st.markdown("### 🤖 Machine Learning")
            st.markdown("""
            ```
            Dados + Resultados → Regras
            ```
            """)
            st.write("As regras são descobertas pelo algoritmo")
        
        # Tipos de aprendizado
        st.subheader("📚 Tipos de Aprendizagens")
        
        tab1, tab2, tab3 = st.tabs(["Supervisionado", "Não Supervisionado", "Por Reforço"])
        
        with tab1:
            st.markdown("### Aprendizagem Supervisionado")
            st.write("""
            Neste tipo de aprendizagem, o sistema **aprende com exemplos que já têm a resposta certa**. 
            Assim, pode depois aplicar esse conhecimento para prever novos casos.

            **Exemplos:**
            - Identificar e-mails como spam ou não spam
            - Prever o valor de uma casa
            - Distinguir transações legítimas de fraudulentas
            """)
            
            # Demonstração visual simples
            st.markdown("🖼️ Exemplo Visual: Transações Legítimas vs Fraudulentas")
            
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Amostra pequena para demonstração
            sample = df.sample(100, random_state=42)
            ax.scatter(sample["Amount"], sample["V1"], c=sample["Class"], cmap="coolwarm", s=50)
            ax.set_xlabel("Valor da Transação")
            ax.set_ylabel("Componente V1")
            ax.set_title("Classificação de Transações")
            
            # Adicionar legenda manual
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', label='Fraude')
            blue_patch = mpatches.Patch(color='blue', label='Legítima')
            ax.legend(handles=[red_patch, blue_patch])
            
            st.pyplot(fig)
        
        with tab2:
            st.markdown("### Aprendizagem Não Supervisionado")
            st.write("""
            Neste caso, o sistema **não sabe as respostas certas**. Ele tenta encontrar **agrupamentos ou padrões escondidos** nos dados por conta própria.

            **Exemplos:**
            - Agrupar clientes com perfis semelhantes
            - Identificar padrões incomuns
            """)
            
            # Demonstração visual de clustering
            st.markdown(" 🖼️ Exemplo Visual: Agrupamento de Transações")
            
            from sklearn.cluster import KMeans
            
            # Amostra para demonstração
            sample = df.sample(200, random_state=42)
            X = sample[["Amount", "V1"]].values
            
            # Aplicar K-means
            kmeans = KMeans(n_clusters=3, random_state=42)
            sample_clusters = kmeans.fit_predict(X)
            
            # Visualizar
            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(X[:, 0], X[:, 1], c=sample_clusters, cmap="viridis", s=50)
            ax.set_xlabel("Valor da Transação")
            ax.set_ylabel("Componente V1")
            ax.set_title("Agrupamento de Transações (K-means)")
            
            # Adicionar centróides
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                      marker='X', s=200, color='red', label='Centróides')
            ax.legend()
            
            st.pyplot(fig)
        
        with tab3:
            st.markdown(" Aprendizagem por Reforço")
            st.write("""
            Aqui, o sistema **aprende por tentativa e erro**. Ele testa ações e **recebe recompensas ou penalizações** com base no que acontece. Com o tempo, aprende quais decisões levam aos melhores resultados.
            """)
            
            st.image("https://cdn-images-1.medium.com/max/800/1*Z2yMvuRTXcMHRdHzKMRM5w.png", 
                    caption="Ciclo de Aprendizado por Reforço", width=400)
    
        # Processo de Machine Learning
        st.subheader("⚙️ Processo de Machine Learning")
    
        process_steps = {
             "1. Preparação de Dados": "Recolha, limpeza, normalização e divisão em conjuntos de treinamento/teste",
             "2. Seleção de Modelo": "Escolha do algoritmo mais adequado para o problema",
             "3. Treinamento": "Ajuste dos parâmetros do modelo usando dados de treinamento",
             "4. Validação": "Avaliação do desempenho em dados não vistos anteriormente",
             "5. Otimização": "Melhoria do modelo ajustando as suas configurações",
             "6. Implementação": "Colocação do modelo em produção",
             "7. Monitorização": "Acompanhamento contínuo do desempenho"
          }
    
        col1, col2 = st.columns(2)
    
        for i, (step, desc) in enumerate(process_steps.items()):
           if i < 4:
              col1.markdown(f"**{step}:** {desc}")
           else:
              col2.markdown(f"**{step}:** {desc}")
    
         # Aplicações em detecção de fraudee
        st.subheader("💳 Como o Machine Learning Ajuda a Detetar Fraudes")
    
        st.write("""
          A deteção de fraudes é uma das aplicações mais valiosas do machine learning no setor financeiro. 
          Estes modelos conseguem identificar comportamentos suspeitos que muitas vezes escapam à análise humana.

          **Vantagens:**
          - ⚡ Análise em tempo real
          - 📈 Capacidade de adaptação a novos tipos de fraude
          - 🎯 Redução de falsos alarmes
          - 🧠 Processamento de grandes volumes de transações

         **Desafios:**
          - ⚖️ Poucas fraudes em comparação com transações normais (desequilíbrio nos dados)
          - 🕵️‍♂️ Novas formas de fraude surgem constantemente
          - ⏱️ Necessidade de decisões rápidas
          - 🔐 Proteção dos dados dos clientes
        """)
    
        # Métricas de avaliação
        st.subheader("📏 Como Avaliamos se um Modelo é Bom?")
    
        metrics = {
         "Accuracy": "Percentagem total de previsões corretas",
         "Precisão": "Entre os casos classificados como fraude, quantos realmente são fraude",
         "Recall (Sensibilidade)": "Entre as fraudes reais, quantas foram detectadas corretamente",
         "F1-Score": "Equilíbrio entre precisão e recall",
         "AUC-ROC": "Capacidade de distinguir entre classes (0.5 = aleatório, 1.0 = perfeito)",
         "Custo de erros": "Impacto financeiro de uma deteção incorreta"
        }
    
        for metric, desc in metrics.items():
          st.markdown(f"**{metric}**: {desc}")
    
        # Demonstração prática
        st.subheader("🧪 Exemplo Simples de Detecção de Fraudes")
    

        st.write("""
          Abaixo mostramos uma simulação de como um modelo pode aprender a distinguir fraudes de transações legítimas.

         ⚠️ Este é apenas um exemplo simples, com poucas variáveis, usado apenas para fins educativos.
         """)
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        
        # Preparar dados (amostra pequena para demonstração rápida)
        sample = df.sample(1000, random_state=42)
        X = sample[["Amount", "V1", "V3", "V4"]].values
        y = sample["Class"].values
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Versão mais robusta para o treinamento com SMOTE
        with st.spinner('Treinando o modelo...'):
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            
            # Verificar se há amostras suficientes para aplicar SMOTE
            try:
                # Verificar o número mínimo de amostras nas classes
                class_counts = np.bincount(y_train)
                min_class_count = class_counts.min()
                
                # Decidir qual método usar com base no número de amostras
                if min_class_count >= 6:  # Para k=5 o SMOTE precisa de pelo menos 6 amostras por classe
                    # SMOTE regular
                    smote = SMOTE(random_state=42)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                elif min_class_count >= 2:  # Se tiver ao menos 2 amostras, usar k=1
                    st.warning("Poucas amostras na classe minoritária. Usando SMOTE com k=1.")
                    smote = SMOTE(k_neighbors=1, random_state=42)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                else:
                    # Se mesmo assim não for possível, usar os dados originais
                    st.warning("Dados muito desbalanceados. Usando dados originais sem SMOTE.")
                    X_train_resampled, y_train_resampled = X_train, y_train
            except Exception as e:
                # Capturar qualquer erro do SMOTE
                st.error(f"Erro ao aplicar SMOTE: {str(e)}. Usando dados originais.")
                X_train_resampled, y_train_resampled = X_train, y_train
        
        # Treinar o modelo com os dados processados
        model.fit(X_train_resampled, y_train_resampled)
        
        # Fazer previsões
        y_pred = model.predict(X_test)
        
        # Avaliar modelo
        st.write("**Accuracy do modelo:**", accuracy_score(y_test, y_pred))
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # Especificamos explicitamente as classes 0 e 1
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
        ax.set_xlabel('Previsto')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusão')
        st.pyplot(fig)
        
        # Relatório de classificação
        st.write("**Relatório de classificação:**")
        st.text(classification_report(y_test, y_pred, zero_division=0))
        
        # Importância das features
        importances = model.feature_importances_
        feature_names = ["Amount", "V1", "V3", "V4"]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(feature_names, importances)
        ax.set_ylabel('Importância')
        ax.set_title('Importância das Features')
        st.pyplot(fig)
        
        st.write("""
         📌 **Nota:** Este exemplo foi simplificado para melhor compreensão. 
         Em situações reais, são utilizados muitos mais dados e técnicas para garantir uma deteção mais precisa e justa.
        """)
    
    with model_tabs[1]:
        # Mover a demonstração de classificação para esta tab
        st.markdown("##  🕵️‍♂️ Classificação para Identificar Fraudes")
        
        # Carregar dados
        df = pd.read_csv("creditcard.csv")
        df = df.dropna()
        
        # Criar variável alvo (Class) desbalanceada
        df["Class"] = df["Class"].astype("category")
        
        # Amostra dos dados
        st.subheader("🔍 Exemplo de Transações")
        st.write("Aqui está uma amostra aleatória dos dados utilizados na análise:")
        st.write(df.sample(10))
        
        # Contagem das classes
        st.subheader("📊 Quantas fraudes temos?")
        class_counts = df["Class"].value_counts()
        st.bar_chart(class_counts)
        
        # Seleção de variáveis
        st.subheader("🧩 Quais variáveis vamos usar?")
        
        all_columns = df.columns.tolist()
        target = "Class"
        features = st.multiselect(
            "Escolha os dados que o modelo deve usar para aprender:",
            options=all_columns,
            default=all_columns[:-1]  # Selecionar todas menos a última (que é a variável alvo)
        )
        
        # Garantir que a variável alvo não esteja entre as features selecionadas
        if target in features:
            features.remove(target)
        
        st.write("✅ Variáveis selecionadas:", features)
        
        # Dividir dados
        X = df[features]
        y = df[target]
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Treinamento do modelo
        st.subheader("⚙️ Escolha o modelo de classificação")
        
        # Selecionar modelo
        model_type = st.selectbox(
            "Modelo:",
            ["Random Forest", "Regressão Logística", "Árvore de Decisão"]
        )

        if model_type == "Random Forest":
            from sklearn import tree

            st.subheader("🌳 Visualização de uma Árvore Individual do Random Forest")
            tree_idx = st.slider("Escolha o índice da árvore para visualizar", 0, len(model.estimators_) - 1, 0)
            fig, ax = plt.subplots(figsize=(16, 6))
            tree.plot_tree(
                model.estimators_[tree_idx],
                feature_names=features,
                class_names=["Legítima", "Fraude"],
                filled=True,
                rounded=True,
                max_depth=3,  # Limite para facilitar a visualização
                fontsize=10,
                ax=ax
            )
            st.pyplot(fig)
            st.write(f"**Árvore exibida:** Estimador {tree_idx} do Random Forest (apenas os 3 primeiros níveis).")
        elif model_type == "Regressão Logística":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=2000, random_state=42)
        else:
            if model_type == "Árvore de Decisão":
                from sklearn import tree

                st.subheader("🌳 Visualização da Árvore")

                # Se o modelo for uma RandomForest, escolher uma das árvores
                if isinstance(model, RandomForestClassifier):
                    st.warning("Estás a visualizar uma árvore individual de uma Random Forest.")
                    tree_idx = st.slider("Escolhe o índice da árvore a visualizar", 0, len(model.estimators_) - 1, 0)
                    tree_to_plot = model.estimators_[tree_idx]
                else:
                    tree_to_plot = model  # Assume que é DecisionTreeClassifier

                fig, ax = plt.subplots(figsize=(16, 6))
                tree.plot_tree(
                    tree_to_plot,
                    feature_names=features,
                    class_names=["Legítima", "Fraude"],
                    filled=True,
                    rounded=True,
                    max_depth=3,
                    fontsize=10,
                    ax=ax
                )
                st.pyplot(fig)
                st.write("**Nota:** Apenas os 3 primeiros níveis da árvore estão visíveis para facilitar a leitura.")

                # Análise do índice Gini
                st.subheader("📊 Análise do Índice Gini dos Nós da Árvore")
                gini_values = tree_to_plot.tree_.impurity
                node_samples = tree_to_plot.tree_.n_node_samples
                gini_df = pd.DataFrame({
                    "Nó": range(len(gini_values)),
                    "Índice Gini": gini_values,
                    "Amostras no Nó": node_samples
                })
                st.write(gini_df.head(10))  # Mostra os 10 primeiros nós

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(gini_df["Nó"], gini_df["Índice Gini"], marker="o")
                ax.set_xlabel("Nó")
                ax.set_ylabel("Índice Gini")
                ax.set_title("Índice Gini ao longo dos nós da árvore")
                st.pyplot(fig)

                st.write("""
                **O índice Gini mede a impureza dos nós:**
                - Valor 0: nó puro (todas as amostras da mesma classe)
                - Valor próximo de 0.5: mistura equilibrada das classes
                """)
        
        # Treinar modelo
        with st.spinner(f'Treinando o modelo ({model_type})...'):
            # Aplicar SMOTE para balancear as classes
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            model.fit(X_train_resampled, y_train_resampled)

            # Defina o número de componentes do PCA
            n_components = st.slider("Nº de componentes do PCA", 2, min(len(features), 20), 5)

            # Treinamento SEM PCA
            start = time.time()
            model_no_pca = RandomForestClassifier(n_estimators=50, random_state=42)
            model_no_pca.fit(X_train, y_train)
            fit_time_no_pca = time.time() - start
            acc_no_pca = model_no_pca.score(X_test, y_test)

            # Treinamento COM PCA
            pca = PCA(n_components=n_components, random_state=42)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            start = time.time()
            model_pca = RandomForestClassifier(n_estimators=50, random_state=42)
            model_pca.fit(X_train_pca, y_train)
            fit_time_pca = time.time() - start
            acc_pca = model_pca.score(X_test_pca, y_test)

            # Exibir resultados
            st.subheader("Comparação: Com vs. Sem PCA")
            results = pd.DataFrame({
                "Accuracy": [acc_no_pca, acc_pca],
                "Tempo de ajuste (s)": [fit_time_no_pca, fit_time_pca]
            }, index=["Sem PCA", "Com PCA"])
            st.write(results)
        
        # Avaliação do modelo
        st.subheader("Avaliação do Modelo")


        # Fazer previsões
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        
        # Matriz de confusão
        st.subheader("Matriz de Confusão")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # Especificamos explicitamente as classes 0 e 1
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
        ax.set_xlabel('Previsto')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusão')
        st.pyplot(fig)
        
        # Relatório de classificação
        st.write("**Relatório de classificação:**")
        st.text(classification_report(y_test, y_pred, zero_division=0))
        
        # Importância das features (apenas para Random Forest)
        if model_type == "Random Forest":
            st.subheader("Importância das Features")
            
            importances = model.feature_importances_
            feature_names = features
            
            # Criar dataframe de importâncias
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importância': importances
            }).sort_values(by="Importância", ascending=False)
            
            st.write(importance_df)
            
            # Gráfico de barras
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=importance_df, x="Importância", y="Feature", ax=ax, hue="Importância", palette="viridis", legend=False)
            ax.set_title("Importância das Features - Random Forest")
            ax.set_xlabel("Features")
            ax.set_ylabel("Importância")
            st.pyplot(fig)

        # Add to model_tabs[1] after the existing classification models
        st.subheader("🚀 Testar Modelos Avançados")

        st.write("""
       ### Experimente diferentes modelos para detetar possíveis fraudes

       Aqui pode comparar os resultados de dois métodos populares que analisam padrões nos dados. Basta ativar e ajustar os parâmetros desejados.
        """)

        # Create columns for the two models
        col1, col2 = st.columns(2)

        with col1:
            st.write("#### AdaBoost")

            run_ada = st.checkbox("Ativar AdaBoost", value=False)
            if run_ada:
                from sklearn.ensemble import AdaBoostClassifier

                # Set parameters
                n_estimators = st.slider("Número de estimadores (AdaBoost)", 50, 300, 100)
                learning_rate = st.slider("Taxa de aprendizagem (AdaBoost)", 0.01, 2.0, 1.0, 0.01)

                with st.spinner("A treinar modelo..."):
                    # Start timing
                    start = time.time()

                    # Create and train model
                    ada_model = AdaBoostClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                    ada_model.fit(X_train_resampled, y_train_resampled)

                    # End timing
                    duration = time.time() - start
                    st.write(f"⏱️ Training time: {duration:.2f} seconds")

                    # Make predictions
                    y_pred_ada = ada_model.predict(X_test)

                    # Calculate metrics
                    ada_accuracy = accuracy_score(y_test, y_pred_ada)
                    ada_precision = precision_score(y_test, y_pred_ada, zero_division=0)
                    ada_recall = recall_score(y_test, y_pred_ada, zero_division=0)
                    ada_f1 = f1_score(y_test, y_pred_ada, zero_division=0)

                    # Display metrics
                    st.metric("Accuracy", f"{ada_accuracy:.4f}")
                    st.metric("Precisão", f"{ada_precision:.4f}")
                    st.metric("Recall", f"{ada_recall:.4f}")
                    st.metric("F1 Score", f"{ada_f1:.4f}")

                    # Display confusion matrix
                    cm_ada = confusion_matrix(y_test, y_pred_ada)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
                    ax.set_xlabel('Previsto')
                    ax.set_ylabel('Real')
                    ax.set_title('Matriz de Confusão - AdaBoost')
                    st.pyplot(fig)

        with col2:
            st.write("#### XGBoost")

            run_xgb = st.checkbox("Ativar XGBoost", value=False)
            if run_xgb:
                import xgboost as xgb

                # Set parameters
                n_estimators_xgb = st.slider("Número de estimadores (XGBoost)", 50, 300, 100)
                max_depth = st.slider("Profundidade máxima (XGBoost)", 3, 10, 6)
                learning_rate_xgb = st.slider("Taxa de aprendizagem (XGBoost)", 0.01, 0.3, 0.1, 0.01)

                with st.spinner("A treinar modelo..."):
                    # Start timing
                    start = time.time()

                    # Create and train model
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=n_estimators_xgb,
                        max_depth=max_depth,
                        learning_rate=learning_rate_xgb,
                        random_state=42
                    )
                    xgb_model.fit(X_train_resampled, y_train_resampled)

                    # End timing
                    duration = time.time() - start
                    st.write(f"⏱️ Tempo de treino: {duration:.2f} segundos")

                    # Make predictions
                    y_pred_xgb = xgb_model.predict(X_test)

                    # Calculate metrics
                    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
                    xgb_precision = precision_score(y_test, y_pred_xgb, zero_division=0)
                    xgb_recall = recall_score(y_test, y_pred_xgb, zero_division=0)
                    xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)

                    # Display metrics
                    st.metric("Accuracy", f"{xgb_accuracy:.4f}")
                    st.metric("Precisão", f"{xgb_precision:.4f}")
                    st.metric("Recall", f"{xgb_recall:.4f}")
                    st.metric("F1 Score", f"{xgb_f1:.4f}")

                    # Display confusion matrix
                    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
                    ax.set_xlabel('Previsto')
                    ax.set_ylabel('Real')
                    ax.set_title('Matriz de Confusão - XGBoost')
                    st.pyplot(fig)

                    # Feature importance for XGBoost
                    fig, ax = plt.subplots(figsize=(8, 6))
                    xgb.plot_importance(xgb_model, ax=ax, max_num_features=10)
                    plt.title("XGBoost Feature Importance")
                    st.pyplot(fig)

        # Compare boosting models if both have been trained
        if run_ada and run_xgb:
            st.subheader("📊 Comparação entre AdaBoost e XGBoost")

            # Create comparison dataframe
            boost_comparison = pd.DataFrame({
                'Modelo': ['AdaBoost', 'XGBoost'],
                'Accuracy': [ada_accuracy, xgb_accuracy],
                'Precisão': [ada_precision, xgb_precision],
                'Recall': [ada_recall, xgb_recall],
                'F1 Score': [ada_f1, xgb_f1]
            })

            st.write(boost_comparison)

            # Plot comparison
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(2)
            width = 0.2
            metrics = ['Accuracy', 'Precisão', 'Recall', 'F1 Score']
            colors = ['blue', 'green', 'red', 'purple']

            for i, metric in enumerate(metrics):
                values = boost_comparison[metric].values
                ax.bar(x + i*width - 0.3, values, width, label=metric, color=colors[i])

            ax.set_xticks(x)
            ax.set_xticklabels(boost_comparison['Modelo'])
            ax.set_ylabel('Valor')
            ax.set_title('Comparação dos Modelos Boosting')
            ax.legend()

            st.pyplot(fig)

        # Add to model_tabs[1] after boosting methods
        st.subheader("🔄 Avaliar Modelos com SVM")

        st.write("""
        Pode também experimentar outro tipo de modelo para comparar resultados. Aqui é possível testar diferentes versões (chamadas **kernels**) e ver qual funciona melhor com os dados.
        """)

        run_svm = st.checkbox("Testar modelos SVM", value=False)

        if run_svm:
            from sklearn.svm import SVC

            # Kernels to test
            kernels = ["linear", "poly", "rbf", "sigmoid"]

            # Create dictionary to store results
            svm_results = {}
            training_times = {}

            with st.spinner("A treinar modelos SVM com diferentes configurações..."):
                for kernel in kernels:
                    # Start timing
                    start = time.time()

                    # Create and train model
                    svm_model = SVC(
                        kernel=kernel,
                        probability=True,
                        random_state=42,
                        class_weight='balanced'
                    )

                    # Fit model
                    svm_model.fit(X_train_resampled, y_train_resampled)

                    # End timing
                    duration = time.time() - start
                    training_times[kernel] = duration

                    # Make predictions
                    y_pred_svm = svm_model.predict(X_test)

                    # Calculate metrics
                    svm_accuracy = accuracy_score(y_test, y_pred_svm)
                    svm_precision = precision_score(y_test, y_pred_svm, zero_division=0)
                    svm_recall = recall_score(y_test, y_pred_svm, zero_division=0)
                    svm_f1 = f1_score(y_test, y_pred_svm, zero_division=0)

                    # Store results
                    svm_results[kernel] = {
                        'Accuracy': svm_accuracy,
                        'Precisão': svm_precision,
                        'Recall': svm_recall,
                        'F1 Score': svm_f1,
                        'Tempo de Treino': duration
                    }

            # Display results as table
            st.subheader("📊 Resultados dos Modelos SVM")
            st.write("Aqui pode ver o desempenho dos diferentes tipos de SVM testados:")
            svm_df = pd.DataFrame.from_dict(svm_results, orient='index')
            st.write(svm_df)
            
            # Find best kernel
            best_kernel = svm_df['F1 Score'].idxmax()
            st.write(f"🟢 Melhor desempenho (F1 Score): **{best_kernel}**")
            
            # Plot performance metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(kernels))
            width = 0.2
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            colors = ['blue', 'green', 'red', 'purple']
            
            for i, metric in enumerate(metrics):
                values = [svm_results[kernel][metric] for kernel in kernels]
                ax.bar(x + i*width - 0.3, values, width, label=metric, color=colors[i])
            
            ax.set_xticks(x)
            ax.set_xticklabels(kernels)
            ax.set_ylabel('Valor')
            ax.set_title('Comparação dos Modelos SVM por Tipo')
            ax.legend()
            
            st.pyplot(fig)
            
            # Plot training times
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(kernels, [training_times[k] for k in kernels], color='teal')
            ax.set_xlabel('Tipo de Kernel')
            ax.set_ylabel('Tempo de Treino (segundos)')
            ax.set_title('Tempo de Treino por Tipo de SVM')
            for i, v in enumerate([training_times[k] for k in kernels]):
                ax.text(i, v + 0.1, f"{v:.2f}s", ha='center')
            st.pyplot(fig)


        # Add to model_tabs[1] after SVM section
        st.subheader("📊 Modelo Naive Bayes")
        run_nb = st.checkbox("Testar Modelo Naive Bayes", value=False)
        
        if run_nb:
            from sklearn.naive_bayes import GaussianNB
            
            with st.spinner("A treinar modelo..."):
                # Start timing
                start = time.time()
                
                # Create and train model
                nb_model = GaussianNB()
                nb_model.fit(X_train_resampled, y_train_resampled)
                
                # End timing
                duration = time.time() - start
                st.write(f"⏱️ Tempo de treino: {duration:.2f} segundos")
                
                # Make predictions
                y_pred_nb = nb_model.predict(X_test)
                
                # Get probabilities
                y_proba_nb = nb_model.predict_proba(X_test)
                
                # Calculate metrics
                nb_accuracy = accuracy_score(y_test, y_pred_nb)
                nb_precision = precision_score(y_test, y_pred_nb, zero_division=0)
                nb_recall = recall_score(y_test, y_pred_nb, zero_division=0)
                nb_f1 = f1_score(y_test, y_pred_nb, zero_division=0)
                
                # Display metrics
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{nb_accuracy:.4f}")
                col1.metric("Precisão", f"{nb_precision:.4f}")
                col2.metric("Recall", f"{nb_recall:.4f}")
                col2.metric("F1 Score", f"{nb_f1:.4f}")
                
                # Display confusion matrix
                cm_nb = confusion_matrix(y_test, y_pred_nb)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Legítima', 'Fraude'], 
                           yticklabels=['Legítima', 'Fraude'])
                ax.set_xlabel('Previsto')
                ax.set_ylabel('Real')
                ax.set_title('Matriz de Confusão - Naive Bayes')
                st.pyplot(fig)
                
                # Classification report
                st.write("### Naive Bayes Classification Report")
                st.text(classification_report(y_test, y_pred_nb))
                
                # Plot probability distribution
                st.write("### Distribuição de Probabilidades de Fraude")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(
                    data=pd.DataFrame({
                        'Probabilidade de Fraude': y_proba_nb[:, 1],
                        'Classe Real': y_test
                    }),
                    x='Probabilidade de Fraude',
                    hue='Classe Real',
                    bins=50,
                    ax=ax
                )
                plt.axvline(x=0.5, color='red', linestyle='--', label='Default threshold (0.5)')
                plt.legend()
                st.pyplot(fig)
                
                # Add custom threshold slider
                st.write("### Ajustar Limiar de Decisão")
                nb_threshold = st.slider("Limiar (threshold)", 0.0, 1.0, 0.5, 0.01)
                
                # Apply custom threshold
                y_pred_custom_nb = (y_proba_nb[:, 1] >= nb_threshold).astype(int)
                
                # Calculate metrics with custom threshold
                custom_accuracy = accuracy_score(y_test, y_pred_custom_nb)
                custom_precision = precision_score(y_test, y_pred_custom_nb, zero_division=0)
                custom_recall = recall_score(y_test, y_pred_custom_nb, zero_division=0)
                custom_f1 = f1_score(y_test, y_pred_custom_nb, zero_division=0)
                
                # Display metrics with custom threshold
                col1, col2 = st.columns(2)
                col1.metric("Accuracy (Limiar)", f"{custom_accuracy:.4f}")
                col1.metric("Precisão (Limiar)", f"{custom_precision:.4f}")
                col2.metric("Recall (Limiar)", f"{custom_recall:.4f}")
                col2.metric("F1 Score (Limiar)", f"{custom_f1:.4f}")
                
                # Display confusion matrix with custom threshold
                cm_custom_nb = confusion_matrix(y_test, y_pred_custom_nb)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm_custom_nb, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Legítima', 'Fraude'], 
                           yticklabels=['Legítima', 'Fraude'])
                ax.set_xlabel('Previsto')
                ax.set_ylabel('Real')
                ax.set_title(f'Matriz de Confusão - Limiar{nb_threshold:.2f}')
                st.pyplot(fig)

        # K-Nearest Neighbors (K-NN)
        st.subheader("🔎 K-Nearest Neighbors (K-NN)")

        run_knn = st.checkbox("Treinar modelo K-NN", value=False)
        if run_knn:
            from sklearn.neighbors import KNeighborsClassifier

            # Seleção do número de vizinhos
            k_range = st.slider("Escolha o intervalo de k (nº de vizinhos)", 1, 20, (3, 10))
            k_values = list(range(k_range[0], k_range[1] + 1))
            f1_scores = []

            with st.spinner("A treinar modelos para diferentes valores de k..."):
                for k in k_values:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_train_resampled, y_train_resampled)
                    y_pred = knn.predict(X_test)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    f1_scores.append(f1)

            # Gráfico F1-Score x k
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(k_values, f1_scores, marker='o')
            ax.set_xlabel("Número de Vizinhos (k)")
            ax.set_ylabel("F1-Score")
            ax.set_title("Desempenho do K-NN")
            st.pyplot(fig)

            # Melhor k
            best_k = k_values[np.argmax(f1_scores)]
            st.write(f"Melhor valor k encontrado: **{best_k}** (F1-Score = {max(f1_scores):.4f})")

            # Avaliação detalhada para o melhor k
            knn_best = KNeighborsClassifier(n_neighbors=best_k)
            knn_best.fit(X_train_resampled, y_train_resampled)
            y_pred_best = knn_best.predict(X_test)
            st.write("**Relatório de classificação (melhor k):**")
            st.text(classification_report(y_test, y_pred_best, zero_division=0))
            cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'],
                        yticklabels=['Legítima', 'Fraude'])
            ax.set_xlabel('Previsto')
            ax.set_ylabel('Real')
            ax.set_title(f'Matriz de Confusão - K-NN (k={best_k})')
            st.pyplot(fig)

        st.subheader("🧠 Rede Neural (MLPClassifier)")

        run_mlp = st.checkbox("Treinar Rede Neural", value=False)
        if run_mlp:
            from sklearn.neural_network import MLPClassifier

            # Parâmetros da rede
            hidden_layer_sizes = st.slider("Neurónios por camada", 5, 100, 20)
            n_layers = st.slider("Número de camadas ocultas", 1, 3, 1)
            alpha = st.slider("Alpha (regularização)", 0.0001, 0.1, 0.001, step=0.0001)
            max_iter = st.slider("Épocas de treinamento (max_iter)", 100, 1000, 300, step=50)

            # Definir arquitetura
            layers = tuple([hidden_layer_sizes] * n_layers)

            with st.spinner("A treinar a rede neural..."):
                mlp = MLPClassifier(hidden_layer_sizes=layers, alpha=alpha, max_iter=max_iter, random_state=42)
                mlp.fit(X_train_resampled, y_train_resampled)
                y_pred_mlp = mlp.predict(X_test)

            # Avaliação
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred_mlp):.4f}")
            st.write(f"F1-Score: {f1_score(y_test, y_pred_mlp, zero_division=0):.4f}")

            # Matriz de confusão
            cm = confusion_matrix(y_test, y_pred_mlp, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'],
                        yticklabels=['Legítima', 'Fraude'])
            ax.set_xlabel('Previsto')
            ax.set_ylabel('Real')
            ax.set_title('Matriz de Confusão - MLPClassifier')
            st.pyplot(fig)

            # Relatório de classificação
            st.write("**Relatório de classificação:**")
            st.text(classification_report(y_test, y_pred_mlp, zero_division=0))


        with st.expander("🚀 Random Forest (substituto ao AutoML)"):
            st.info("Modelo alternativo ao AutoML, compatível com a versão usada.")

            run_rf = st.checkbox("Executar Random Forest", value=False)

            if run_rf:
                with st.spinner("A treinar modelo Random Forest..."):
                    from sklearn.metrics import classification_report, accuracy_score

                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train_resampled, y_train_resampled)

                    y_pred_rf = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred_rf)
                    st.write(f"**Accuracy Random Forest:** {accuracy:.4f}")

                    st.text("Relatório de Classificação - Random Forest")
                    st.text(classification_report(y_test, y_pred_rf, zero_division=0))

                    st.subheader("🏆 Importância das Features")
                    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    feature_importances.nlargest(10).plot(kind='barh', ax=ax)
                    ax.set_title("Top 10 Features mais importantes")
                    st.pyplot(fig)

    with model_tabs[2]:
        st.markdown("## 🔍 Análise com Ridge e Lasso")

        st.write("""
       Vamos explorar diferentes formas de prever se uma transação é fraudulenta ou não.  
    Os modelos apresentados utilizam combinações de variáveis para fazer essa previsão, ajudando-nos a compreender quais são mais relevantes.
        """)
        
        # Seleção de variáveis
        st.subheader("⚙️ Configurações Iniciais")

        # A variável alvo agora é fixa como "Class"
        target_column = "Class"
        st.write(f"**Variável alvo:** {target_column} (0: Legítima, 1: Fraudulenta)")
        
        n_features = st.slider("Número de features a utilizar", 2, 10, 5)
        
        # Seleção automática de features mais correlacionadas com a variável Class
        numeric_df = df.select_dtypes(include=['number'])
        if target_column in numeric_df.columns:
            correlations = numeric_df.drop(columns=[target_column]).corrwith(df[target_column]).abs().sort_values(ascending=False)
        else:
            correlations = numeric_df.corrwith(df[target_column]).abs().sort_values(ascending=False)
        best_features = correlations[:n_features].index.tolist()
        
        st.write(f"Variáveis selecionadas (mais associadas com fraudes {target_column}):")
        st.write(best_features)
        
        # Dividir dados
        X = df[best_features].values
        y = df[target_column].values
        
        # Normalizar dados
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir em treino e teste
        test_size = st.slider("Percentagem de dados para teste", 10, 40, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)
        
        # Configuração dos modelos
        st.subheader("🔧 Parâmetros dos Modelos")
        
        col1, col2 = st.columns(2)
        with col1:
            alpha_ridge = st.slider(
                "Ajuste do modelo Ridge",
                0.01, 10.0, 1.0, 0.01
            )
    
        with col2:
            alpha_lasso = st.slider(
                "Ajuste do modelo Lasso",
                0.001, 1.0, 0.01, 0.001
            )
    
        # Treinamento dos modelos
        with st.spinner("A treinar modelos..."):
            # Linear Regression (sem regularização)
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            
            # Ridge Regression
            ridge = Ridge(alpha=alpha_ridge)
            ridge.fit(X_train, y_train)
            
            # Lasso Regression
            lasso = Lasso(alpha=alpha_lasso, max_iter=10000)
            lasso.fit(X_train, y_train)
        
        # Avaliação dos modelos
        models = {
            "Regressão Linear": lr,
            f"Ridge (α={alpha_ridge})": ridge,
            f"Lasso (α={alpha_lasso})": lasso
        }
        
        # Configurar um limiar para converter previsões contínuas em binárias
        threshold = 0.5
        
        results = {}
        predictions = {}
        
        for name, model in models.items():
            # Previsões contínuas
            y_pred_proba = model.predict(X_test)
            # Converter para binárias usando threshold
            y_pred_binary = (y_pred_proba > threshold).astype(int)
            predictions[name] = y_pred_binary
            
            # Métricas de classificação
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary, zero_division=0)
            recall = recall_score(y_test, y_pred_binary, zero_division=0)
            f1 = f1_score(y_test, y_pred_binary, zero_division=0)
            mse = mean_squared_error(y_test, y_pred_proba)
            
            results[name] = {
                "Accuracy": accuracy,
                "Precisão": precision,
                "Recall": recall, 
                "F1-Score": f1,
                "Erro Médio (MSE)": mse
            }
        
        # Mostrar resultados
        st.subheader("📊 Comparação dos Resultados")
        
        # Criar dataframe de resultados
        results_df = pd.DataFrame({
            model: metrics
            for model, metrics in results.items()
        }).T
        
        st.write(results_df)
        
        # Gráfico de barras para F1-Score (melhor métrica para dados desbalanceados)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results_df.index, results_df["F1-Score"], color=["blue", "green", "orange"])
        ax.set_ylabel('F1-Score por Modelo')
        ax.set_title('F1-Score (quanto maior, melhor)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Visualizar coeficientes
        st.subheader("📌 Relevância das Variáveis")
        
        coef_df = pd.DataFrame({
            'Variável': best_features,
            'Linear Regression': lr.coef_,
            f'Ridge (α={alpha_ridge})': ridge.coef_,
            f'Lasso (α={alpha_lasso})': lasso.coef_
        })
        
        st.write(coef_df.set_index('Variável'))
        
        # Gráfico de coeficientes
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.25
        index = np.arange(len(best_features))
        
        # Plotar barras para cada modelo
        ax.bar(index - bar_width, lr.coef_, bar_width, label='Linear Regression', color='blue')
        ax.bar(index, ridge.coef_, bar_width, label=f'Ridge (α={alpha_ridge})', color='green')
        ax.bar(index + bar_width, lasso.coef_, bar_width, label=f'Lasso (α={alpha_lasso})', color='orange')
        
        # Adicionar linha zero para referência
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Configurar labels e legendas
        ax.set_xlabel('Variável')
        ax.set_ylabel('Coeficientes')
        ax.set_title('Importância das Variável para Detecção de Fraudes')
        ax.set_xticks(index)
        ax.set_xticklabels(best_features, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)

    
        # Adicionar thresholding interativo
        st.subheader("🎯 Ajuste de Sensibilidade (Threshold)")
        
        st.write("""
        Pode ajustar a sensibilidade da decisão do modelo.  
    Um valor mais baixo pode detetar mais fraudes (mas com mais falsos positivos),  
    enquanto um valor mais alto é mais cauteloso.
        """)
        
        # Escolher um modelo para ajustar o threshold
        model_for_threshold = st.selectbox(
            "Escolha um modelo para ajustar o limiar:",
            list(models.keys())
        )
        
        # Obter as previsões contínuas
        selected_model = models[model_for_threshold]
        y_scores = selected_model.predict(X_test)
        
        # Slider para threshold
        custom_threshold = st.slider(
            "Limiar de decisão",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )
        
        # Aplicar threshold
        y_pred_custom = (y_scores > custom_threshold).astype(int)
        
        # Calcular métricas com threshold personalizado
        custom_accuracy = accuracy_score(y_test, y_pred_custom)
        custom_precision = precision_score(y_test, y_pred_custom, zero_division=0)
        custom_recall = recall_score(y_test, y_pred_custom, zero_division=0)
        custom_f1 = f1_score(y_test, y_pred_custom, zero_division=0)
        
        # Exibir métricas com threshold personalizado
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{custom_accuracy:.4f}")
        col1.metric("Precisão", f"{custom_precision:.4f}")
        col2.metric("Recall", f"{custom_recall:.4f}")
        col2.metric("F1-Score", f"{custom_f1:.4f}")
        
        # Matriz de confusão com threshold personalizado
        cm_custom = confusion_matrix(y_test, y_pred_custom, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
        ax.set_xlabel('Previsto')
        ax.set_ylabel('Real')
        ax.set_title(f'Matriz de Confusão - {model_for_threshold} (Limiar = {custom_threshold})')
        st.pyplot(fig)

elif page == "🧪 Classificar Transação":
    st.markdown('<p class="big-font">🧪 Classificar Transação</p>', unsafe_allow_html=True)
    st.write(" Introduza os dados de uma nova transação para saber se ela poderá ser considerada suspeita. O sistema analisa os valores com base em padrões reais já observados em transações anteriores.")

    # Features usadas no modelo
    selected_features = ["Amount", "V1", "V2", "V3", "V4", "V10"]

    # Inputs do utilizador
    input_data = {}
    for feature in selected_features:
        input_data[feature] = st.number_input(f"Insera valor para: {feature}", value=0.0)

    # Criar DataFrame a partir dos inputs
    input_df = pd.DataFrame([input_data])

    # Preparar os dados de treino
    X = df[selected_features]
    y = df["Class"]

    # Normalizar com StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Treinar modelo de regressão logística
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    model.fit(X_scaled, y)

    # Transformar input do utilizador
    input_scaled = scaler.transform(input_df)

    # Prever probabilidade de fraude
    prob_fraude = model.predict_proba(input_scaled)[0][1]

    st.write(f"🔍 **Probabilidade de ser fraude:** {prob_fraude:.4f}")

    # Definir limiar
    limiar = 0.5
    if prob_fraude > limiar:
        st.error("🚨 Transação suspeita de fraude!")
    else:
        st.success("✅ Transação legítima.")
