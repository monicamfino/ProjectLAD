import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

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
    "🧭 Dados",
    "🤖 Machine Learning"
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
    sub_page = st.sidebar.radio("Subtópicos", ["📑 Gerar Relatório", "⚙ Configurações Avançadas", "🔄 Normalização e Padronização"])

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

    # 🔄 Normalização e Padronização
    elif sub_page == "🔄 Normalização e Padronização":
        st.subheader("🔄 Padronização e Normalização de Dados")
        
        st.write("""
        ## Padronização (Standardization)

        A padronização (Z-score normalization) é uma técnica de pré-processamento de dados que transforma os valores 
        para que tenham média 0 e desvio padrão 1.
        """)

        # Fórmula matemática com LaTeX
        st.latex(r'Z = \frac{X - \mu}{\sigma}')
        
        st.write("""
        onde:
        - X = valor original
        - μ = média da distribuição 
        - σ = desvio padrão da distribuição
        
        **Características:**
        - Resulta em dados com média 0
        - Resulta em dados com desvio padrão 1
        - Útil quando os dados seguem distribuição normal
        - Preserva outliers (valores extremos)
        
        **Vantagens:**
        - Facilita a comparação entre diferentes atributos
        - Essencial para algoritmos sensíveis à escala (como SVM, K-means, PCA)
        - Melhora a convergência em algoritmos de gradient descent
        """)
        
        # Demonstração de padronização com os dados
        with st.expander("🔍 Demonstração de Padronização"):
            # Selecionar uma coluna para demonstração
            selected_column = st.selectbox("Selecione uma coluna para padronização:", 
                                          df.select_dtypes(include=['number']).columns)
            
            # Calcular média e desvio padrão
            mean_value = df[selected_column].mean()
            std_value = df[selected_column].std()
            
            # Criar uma amostra de dados padronizados
            original_data = df[selected_column].head(10).values
            standardized_data = (original_data - mean_value) / std_value
            
            # Mostrar uma comparação
            comparison_df = pd.DataFrame({
                "Original": original_data,
                "Padronizado": standardized_data
            })
            
            st.write("**Dados Originais vs. Padronizados:**")
            st.write(comparison_df)
            
            # Mostrar estatísticas
            st.write(f"**Média Original:** {mean_value:.4f}")
            st.write(f"**Desvio Padrão Original:** {std_value:.4f}")
            st.write(f"**Média dos Dados Padronizados:** {standardized_data.mean():.4f}")
            st.write(f"**Desvio Padrão dos Dados Padronizados:** {standardized_data.std():.4f}")
            
            # Plotar comparação
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].hist(original_data, bins=10, color='blue', alpha=0.7)
            ax[0].set_title("Dados Originais")
            ax[1].hist(standardized_data, bins=10, color='green', alpha=0.7)
            ax[1].set_title("Dados Padronizados")
            st.pyplot(fig)
        
        st.write("""
        ## Normalização (Min-Max Scaling)

        A normalização transforma os dados para um intervalo específico, tipicamente [0,1] ou [-1,1].
        """)
        
        # Fórmula matemática com LaTeX
        st.latex(r"X' = \frac{X - X_{min}}{X_{max} - X_{min}}")
        
        st.write("""
        onde:
        - X = valor original
        - Xmin = valor mínimo do atributo
        - Xmax = valor máximo do atributo
        
        **Características:**
        - Escala os dados para um intervalo fixo
        - Preserva a distribuição original dos dados
        - Útil quando a distribuição não é gaussiana
        - Mantém relações entre valores originais
        
        **Vantagens:**
        - Facilita comparação entre variáveis de unidades diferentes
        - Útil para algoritmos que exigem valores limitados
        - Boa para técnicas como redes neurais e algoritmos baseados em distância
        """)
        
        # Demonstração de normalização com os dados
        with st.expander("🔍 Demonstração de Normalização"):
            # Selecionar uma coluna para demonstração
            selected_column = st.selectbox("Selecione uma coluna para normalização:", 
                                          df.select_dtypes(include=['number']).columns,
                                          key="normalization_column")
            
            # Calcular min e max
            min_value = df[selected_column].min()
            max_value = df[selected_column].max()
            
            # Criar uma amostra de dados normalizados
            original_data = df[selected_column].head(10).values
            normalized_data = (original_data - min_value) / (max_value - min_value)
            
            # Mostrar uma comparação
            comparison_df = pd.DataFrame({
                "Original": original_data,
                "Normalizado": normalized_data
            })
            
            st.write("**Dados Originais vs. Normalizados:**")
            st.write(comparison_df)
            
            # Mostrar estatísticas
            st.write(f"**Valor Mínimo Original:** {min_value:.4f}")
            st.write(f"**Valor Máximo Original:** {max_value:.4f}")
            st.write(f"**Valor Mínimo Normalizado:** {normalized_data.min():.4f}")
            st.write(f"**Valor Máximo Normalizado:** {normalized_data.max():.4f}")
            
            # Plotar comparação
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].hist(original_data, bins=10, color='blue', alpha=0.7)
            ax[0].set_title("Dados Originais")
            ax[1].hist(normalized_data, bins=10, color='red', alpha=0.7)
            ax[1].set_title("Dados Normalizados")
            st.pyplot(fig)
        
        st.write("""
        ## Quando Usar Cada Técnica
        
        **Use Padronização quando:**
        - Os dados seguem distribuição normal ou próxima dela
        - O algoritmo pressupõe normalidade dos dados
        - Há presença significativa de outliers que não devem ser ocultados
        - Trabalhando com algoritmos como SVM, regressão linear, ou PCA
        
        **Use Normalização quando:**
        - Precisa de um intervalo específico e limitado
        - Trabalhando com redes neurais, especialmente com funções de ativação que esperam entradas em [0,1] ou [-1,1]
        - A distribuição dos dados não é gaussiana
        - A escala absoluta é importante para o algoritmo
        
        ## Importância no Big Data
        
        - Permite comparabilidade entre diferentes fontes de dados
        - Reduz o impacto de diferentes magnitudes entre variáveis
        - Essencial para algoritmos de aprendizado de máquina que são sensíveis à escala
        - Melhora a qualidade dos resultados de clustering e classificação
        - Facilita a integração de dados heterogêneos
        """)
        
        # Aplicação prática
        st.subheader("🧪 Aplicação Prática")
        
        st.write("""
        Exemplo prático de como a padronização e normalização podem afetar a detecção de fraudes:
        
        Considere as variáveis 'Amount' e 'Time' que possuem escalas muito diferentes. Um algoritmo de detecção de fraude 
        baseado em distância (como KNN) daria peso desproporcional à variável com maior magnitude. Ao normalizar ou 
        padronizar, ambas as variáveis têm peso equivalente na decisão do algoritmo.
        """)
        
        # Comparação visual final
        st.subheader("📊 Comparação Visual")
        
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
        ax.set_title("Dados Originais")
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

    # Adicionar scope das variáveis
    st.subheader("📄 Scope das Variáveis")

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
    st.subheader("📊 Gráficos de Valores Mínimo e Máximo")

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
    st.subheader("📊 Gráficos de Valores Mínimo e Máximo")

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
    ax.set_title("Valores Mínimo e Máximo das Variáveis Especificadas")
    ax.set_ylabel("Valores")
    ax.legend()
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Adicionar legenda explicativa
    st.markdown("""
    **Legenda:**
    - **Min**: O valor mínimo registrado para a variável.
    - **Max**: O valor máximo registrado para a variável.
    Estes valores ajudam a entender a amplitude e a variação dos dados para cada variável.
    """)

# Nova página: Machine Learning
elif page == "🤖 Machine Learning":
    # Adicionar tabs para diferentes modelos de ML
    model_tabs = st.tabs(["Introdução", "Classificação", "Ridge e Lasso Regression"])
    
    with model_tabs[0]:
        # Mover o conteúdo existente sobre ML para esta tab
        st.markdown("## Introdução ao Machine Learning")
        
        # Conceitos básicos
        st.subheader("🔍 Conceitos Básicos")
        st.write("""
        **Machine Learning (ML)** é um subcampo da Inteligência Artificial que permite aos computadores aprender 
        sem programação explícita. Ao contrário da programação tradicional onde escrevemos regras específicas, 
        no ML os algoritmos aprendem padrões diretamente a partir dos dados.
        
        A principal diferença é que em ML:
        - Os dados ensinam o computador
        - O sistema melhora com a experiência
        - Identifica padrões estatisticamente significativos
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
        st.subheader("📚 Tipos de Aprendizado")
        
        tab1, tab2, tab3 = st.tabs(["Supervisionado", "Não Supervisionado", "Por Reforço"])
        
        with tab1:
            st.markdown("### Aprendizado Supervisionado")
            st.write("""
            No aprendizado supervisionado, o algoritmo é treinado em um conjunto de dados rotulado, 
            onde para cada exemplo temos uma entrada e a saída desejada.
            
            **Exemplos de aplicações:**
            - Classificação de e-mails em spam ou não-spam
            - Previsão de preços de imóveis
            - Diagnóstico médico
            
            **Algoritmos populares:**
            - Regressão Linear/Logística
            - Árvores de Decisão
            - Random Forests
            - Support Vector Machines (SVM)
            - Redes Neurais
            """)
            
            # Demonstração visual simples
            st.markdown("#### Exemplo: Classificação de Fraudes")
            
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Amostra pequena para demonstração
            sample = df.sample(100, random_state=42)
            ax.scatter(sample["Amount"], sample["V1"], c=sample["Class"], cmap="coolwarm", s=50)
            ax.set_xlabel("Valor da Transação")
            ax.set_ylabel("Componente V1")
            ax.set_title("Exemplo de Classificação: Transações Legítimas vs Fraudulentas")
            
            # Adicionar legenda manual
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', label='Fraude')
            blue_patch = mpatches.Patch(color='blue', label='Legítima')
            ax.legend(handles=[red_patch, blue_patch])
            
            st.pyplot(fig)
        
        with tab2:
            st.markdown("### Aprendizado Não Supervisionado")
            st.write("""
            No aprendizado não supervisionado, o algoritmo trabalha com dados não rotulados, 
            buscando encontrar estruturas ou padrões intrínsecos nos dados.
            
            **Exemplos de aplicações:**
            - Segmentação de clientes
            - Agrupamento de notícias semelhantes
            - Detecção de anomalias
            - Redução de dimensionalidade
            
            **Algoritmos populares:**
            - K-means
            - DBSCAN
            - Hierarchical Clustering
            - PCA (Principal Component Analysis)
            - t-SNE
            """)
            
            # Demonstração visual de clustering
            st.markdown("#### Exemplo: Clustering de Transações")
            
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
            ax.set_title("Clustering de Transações (K-means, k=3)")
            
            # Adicionar centróides
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                      marker='X', s=200, color='red', label='Centróides')
            ax.legend()
            
            st.pyplot(fig)
        
        with tab3:
            st.markdown("### Aprendizado por Reforço")
            st.write("""
            No aprendizado por reforço, o algoritmo aprende a tomar decisões interagindo com um ambiente,
            recebendo recompensas ou penalizações pelas ações tomadas.
            
            **Exemplos de aplicações:**
            - Jogos (AlphaGo, Atari)
            - Robótica
            - Sistemas de recomendação
            - Trading automatizado
            
            **Algoritmos populares:**
            - Q-Learning
            - Deep Q-Network (DQN)
            - Policy Gradient
            - Actor-Critic
            """)
            
            st.image("https://cdn-images-1.medium.com/max/800/1*Z2yMvuRTXcMHRdHzKMRM5w.png", 
                    caption="Ciclo de Aprendizado por Reforço", width=400)
    
    # Processo de Machine Learning
    st.subheader("⚙️ Processo de Machine Learning")
    
    process_steps = {
        "1. Preparação de Dados": "Coleta, limpeza, normalização e divisão em conjuntos de treinamento/teste",
        "2. Seleção de Modelo": "Escolha do algoritmo mais adequado para o problema",
        "3. Treinamento": "Ajuste dos parâmetros do modelo usando dados de treinamento",
        "4. Validação": "Avaliação do desempenho em dados não vistos anteriormente",
        "5. Ajuste de Hiperparâmetros": "Otimização do modelo para melhorar o desempenho",
        "6. Implantação": "Colocação do modelo em produção",
        "7. Monitoramento": "Acompanhamento contínuo do desempenho"
    }
    
    col1, col2 = st.columns(2)
    
    for i, (step, desc) in enumerate(process_steps.items()):
        if i < 4:
            col1.markdown(f"**{step}:** {desc}")
        else:
            col2.markdown(f"**{step}:** {desc}")
    
    # Aplicações em detecção de fraude
    st.subheader("💳 Machine Learning na Detecção de Fraudes")
    
    st.write("""
    A detecção de fraudes é uma das aplicações mais importantes de machine learning no setor financeiro.
    Algoritmos ML podem identificar padrões suspeitos e anomalias que seriam difíceis de detectar manualmente.
    
    **Benefícios:**
    
    - **Processamento em tempo real**: análise de transações à medida que ocorrem
    - **Adaptabilidade**: aprendizado contínuo com novos padrões de fraude
    - **Redução de falsos positivos**: melhoria na precisão da detecção
    - **Escalabilidade**: capacidade de processar milhões de transações
    
    **Desafios:**
    
    - **Dados desbalanceados**: geralmente há muito mais transações legítimas que fraudulentas
    - **Adaptação a novas fraudes**: fraudadores evoluem constantemente suas técnicas
    - **Latência**: necessidade de respostas em milissegundos
    - **Dados sensíveis**: questões de privacidade e segurança
    """)
    
    # Métricas de avaliação
    st.subheader("📏 Métricas de Avaliação em Detecção de Fraudes")
    
    metrics = {
        "Acurácia": "Porcentagem total de previsões corretas",
        "Precisão": "Entre os casos classificados como fraude, quantos realmente são fraude",
        "Recall (Sensibilidade)": "Entre as fraudes reais, quantas foram detectadas corretamente",
        "F1-Score": "Média harmônica entre precisão e recall",
        "AUC-ROC": "Capacidade de distinguir entre classes (0.5 = aleatório, 1.0 = perfeito)",
        "Custo de classificação errada": "Perda financeira devido a falsos positivos e falsos negativos"
    }
    
    for metric, desc in metrics.items():
        st.markdown(f"**{metric}**: {desc}")
    
    # Demonstração prática
    st.subheader("🧪 Demonstração Prática")
    
    with st.expander("Clique para ver uma demonstração simplificada de detecção de fraudes"):
        st.write("""
        Abaixo está um exemplo simplificado de como um modelo de classificação pode ser usado para detectar fraudes.
        
        Este exemplo usa apenas duas variáveis para facilitar a visualização, mas modelos reais usariam múltiplas variáveis.
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
        st.write("**Acurácia do modelo:**", accuracy_score(y_test, y_pred))
        
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
        **Observação:** Este é apenas um exemplo simplificado para fins educativos. 
        Em cenários reais, seriam necessários:
        - Pré-processamento mais extenso dos dados
        - Utilização de mais features
        - Ajuste de hiperparâmetros
        - Técnicas para lidar com dados desbalanceados
        - Validação cruzada
        """)
    
    with model_tabs[1]:
        # Mover a demonstração de classificação para esta tab
        st.markdown("## Classificação para Detecção de Fraudes")
        
        # Carregar dados
        df = pd.read_csv("creditcard.csv")
        df = df.dropna()
        
        # Criar variável alvo (Class) desbalanceada
        df["Class"] = df["Class"].astype("category")
        
        # Amostra dos dados
        st.subheader("Amostra dos Dados")
        st.write(df.sample(10))
        
        # Contagem das classes
        st.subheader("Distribuição das Classes")
        class_counts = df["Class"].value_counts()
        st.bar_chart(class_counts)
        
        # Seleção de variáveis
        st.subheader("Seleção de Variáveis")
        
        all_columns = df.columns.tolist()
        target = "Class"
        features = st.multiselect(
            "Selecione as variáveis independentes (features):",
            options=all_columns,
            default=all_columns[:-1]  # Selecionar todas menos a última (que é a variável alvo)
        )
        
        # Garantir que a variável alvo não esteja entre as features selecionadas
        if target in features:
            features.remove(target)
        
        st.write("Features selecionadas:", features)
        
        # Dividir dados
        X = df[features]
        y = df[target]
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Treinamento do modelo
        st.subheader("Treinamento do Modelo")
        
        # Selecionar modelo
        model_type = st.selectbox(
            "Escolha o tipo de modelo:",
            ["Random Forest", "Regressão Logística", "Árvore de Decisão"]
        )
        
        if model_type == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(class_weight='balanced', random_state=42)
        elif model_type == "Regressão Logística":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=2000, random_state=42)
        else:
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(random_state=42)
        
        # Treinar modelo
        with st.spinner(f'Treinando o modelo ({model_type})...'):
            # Aplicar SMOTE para balancear as classes
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            model.fit(X_train_resampled, y_train_resampled)
        
        # Avaliação do modelo
        st.subheader("Avaliação do Modelo")
        
        # Fazer previsões
        y_pred = model.predict(X_test)
        
        # Acurácia
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Acurácia: {accuracy:.2f}")
        
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
        st.subheader("🚀 Boosting Methods")
        
        st.write("""
        ### Ensemble Methods: Boosting
        
        Boosting algorithms build multiple models sequentially, with each model correcting 
        the errors of its predecessors. These are particularly effective for fraud detection.
        
        - **AdaBoost**: Adaptive Boosting focuses on misclassified instances by increasing their weights
        - **XGBoost**: eXtreme Gradient Boosting uses gradient descent to minimize errors
        """)
        
        # Create columns for the two models
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### AdaBoost")
            
            run_ada = st.checkbox("Train AdaBoost Model", value=False)
            if run_ada:
                from sklearn.ensemble import AdaBoostClassifier
                
                # Set parameters
                n_estimators = st.slider("Number of Estimators (AdaBoost)", 50, 300, 100)
                learning_rate = st.slider("Learning Rate (AdaBoost)", 0.01, 2.0, 1.0, 0.01)
                
                with st.spinner("Training AdaBoost model..."):
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
                    st.metric("Precision", f"{ada_precision:.4f}")
                    st.metric("Recall", f"{ada_recall:.4f}")
                    st.metric("F1 Score", f"{ada_f1:.4f}")
                    
                    # Display confusion matrix
                    cm_ada = confusion_matrix(y_test, y_pred_ada)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Legitimate', 'Fraud'], 
                               yticklabels=['Legitimate', 'Fraud'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('AdaBoost Confusion Matrix')
                    st.pyplot(fig)
        
        with col2:
            st.write("#### XGBoost")
            
            run_xgb = st.checkbox("Train XGBoost Model", value=False)
            if run_xgb:
                import xgboost as xgb
                
                # Set parameters
                n_estimators_xgb = st.slider("Number of Estimators (XGBoost)", 50, 300, 100)
                max_depth = st.slider("Max Depth (XGBoost)", 3, 10, 6)
                learning_rate_xgb = st.slider("Learning Rate (XGBoost)", 0.01, 0.3, 0.1, 0.01)
                
                with st.spinner("Training XGBoost model..."):
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
                    st.write(f"⏱️ Training time: {duration:.2f} seconds")
                    
                    # Make predictions
                    y_pred_xgb = xgb_model.predict(X_test)
                    
                    # Calculate metrics
                    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
                    xgb_precision = precision_score(y_test, y_pred_xgb, zero_division=0)
                    xgb_recall = recall_score(y_test, y_pred_xgb, zero_division=0)
                    xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
                    
                    # Display metrics
                    st.metric("Accuracy", f"{xgb_accuracy:.4f}")
                    st.metric("Precision", f"{xgb_precision:.4f}")
                    st.metric("Recall", f"{xgb_recall:.4f}")
                    st.metric("F1 Score", f"{xgb_f1:.4f}")
                    
                    # Display confusion matrix
                    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Legitimate', 'Fraud'], 
                               yticklabels=['Legitimate', 'Fraud'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('XGBoost Confusion Matrix')
                    st.pyplot(fig)
                    
                    # Feature importance for XGBoost
                    fig, ax = plt.subplots(figsize=(8, 6))
                    xgb.plot_importance(xgb_model, ax=ax, max_num_features=10)
                    plt.title("XGBoost Feature Importance")
                    st.pyplot(fig)
        
        # Compare boosting models if both have been trained
        if run_ada and run_xgb:
            st.subheader("Comparison of Boosting Methods")
            
            # Create comparison dataframe
            boost_comparison = pd.DataFrame({
                'Model': ['AdaBoost', 'XGBoost'],
                'Accuracy': [ada_accuracy, xgb_accuracy],
                'Precision': [ada_precision, xgb_precision],
                'Recall': [ada_recall, xgb_recall],
                'F1 Score': [ada_f1, xgb_f1]
            })
            
            st.write(boost_comparison)
            
            # Plot comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(2)
            width = 0.2
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            colors = ['blue', 'green', 'red', 'purple']
            
            for i, metric in enumerate(metrics):
                values = boost_comparison[metric].values
                ax.bar(x + i*width - 0.3, values, width, label=metric, color=colors[i])
            
            ax.set_xticks(x)
            ax.set_xticklabels(boost_comparison['Model'])
            ax.set_ylabel('Score')
            ax.set_title('Boosting Methods Comparison')
            ax.legend()
            
            st.pyplot(fig)
        
                # Add to model_tabs[1] after boosting methods
        st.subheader("🔄 Support Vector Machines")
        
        st.write("""
        ### SVM with Multiple Kernels
        
        Support Vector Machines (SVMs) are powerful classifiers that work by finding the hyperplane 
        that best separates classes in the feature space. Different kernels allow SVMs to handle 
        both linear and non-linear classification tasks.
        """)
        
        run_svm = st.checkbox("Train SVM Models with Different Kernels", value=False)
        
        if run_svm:
            from sklearn.svm import SVC
            
            # Kernels to test
            kernels = ["linear", "poly", "rbf", "sigmoid"]
            
            # Create dictionary to store results
            svm_results = {}
            training_times = {}
            
            with st.spinner("Training SVM models with different kernels - this may take a while..."):
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
                        'Precision': svm_precision,
                        'Recall': svm_recall,
                        'F1 Score': svm_f1,
                        'Training Time': duration
                    }
            
            # Display results as table
            st.write("### SVM Performance by Kernel Type")
            svm_df = pd.DataFrame.from_dict(svm_results, orient='index')
            st.write(svm_df)
            
            # Find best kernel
            best_kernel = svm_df['F1 Score'].idxmax()
            st.write(f"Best kernel based on F1 Score: **{best_kernel}**")
            
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
            ax.set_ylabel('Score')
            ax.set_title('SVM Performance by Kernel Type')
            ax.legend()
            
            st.pyplot(fig)
            
            # Plot training times
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(kernels, [training_times[k] for k in kernels], color='teal')
            ax.set_xlabel('Kernel Type')
            ax.set_ylabel('Training Time (seconds)')
            ax.set_title('SVM Training Time by Kernel')
            for i, v in enumerate([training_times[k] for k in kernels]):
                ax.text(i, v + 0.1, f"{v:.2f}s", ha='center')
            st.pyplot(fig)
            
            st.write("""
            ### SVM Kernel Explanation
            
            - **Linear Kernel**: Effective when data is linearly separable. Fast but less flexible.
            - **Polynomial Kernel**: Can model curved decision boundaries with polynomial functions.
            - **RBF (Radial Basis Function)**: Creates complex, non-linear decision boundaries. Often best for diverse data.
            - **Sigmoid Kernel**: Similar to a neural network activation function.
            For fraud detection, RBF kernel typically performs best as fraud patterns are rarely linear.
            """)

                # Add to model_tabs[1] after SVM section
        st.subheader("📊 Naive Bayes Classifier")
        
        st.write("""
        ### Naive Bayes Classification
        
        Naive Bayes is a probabilistic classifier based on Bayes' theorem with an assumption of 
        independence between features. Despite this "naive" assumption, it works surprisingly 
        well in many real-world situations, including fraud detection.
        """)
        
        run_nb = st.checkbox("Train Naive Bayes Model", value=False)
        
        if run_nb:
            from sklearn.naive_bayes import GaussianNB
            
            with st.spinner("Training Naive Bayes model..."):
                # Start timing
                start = time.time()
                
                # Create and train model
                nb_model = GaussianNB()
                nb_model.fit(X_train_resampled, y_train_resampled)
                
                # End timing
                duration = time.time() - start
                st.write(f"⏱️ Training time: {duration:.2f} seconds")
                
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
                col1.metric("Precision", f"{nb_precision:.4f}")
                col2.metric("Recall", f"{nb_recall:.4f}")
                col2.metric("F1 Score", f"{nb_f1:.4f}")
                
                # Display confusion matrix
                cm_nb = confusion_matrix(y_test, y_pred_nb)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Legitimate', 'Fraud'], 
                           yticklabels=['Legitimate', 'Fraud'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Naive Bayes Confusion Matrix')
                st.pyplot(fig)
                
                # Classification report
                st.write("### Naive Bayes Classification Report")
                st.text(classification_report(y_test, y_pred_nb))
                
                # Plot probability distribution
                st.write("### Fraud Probability Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(
                    data=pd.DataFrame({
                        'Fraud Probability': y_proba_nb[:, 1],
                        'Actual Class': y_test
                    }),
                    x='Fraud Probability',
                    hue='Actual Class',
                    bins=50,
                    ax=ax
                )
                plt.axvline(x=0.5, color='red', linestyle='--', label='Default threshold (0.5)')
                plt.legend()
                st.pyplot(fig)
                
                # Add custom threshold slider
                st.write("### Custom Threshold Adjustment")
                nb_threshold = st.slider("Decision Threshold for Naive Bayes", 0.0, 1.0, 0.5, 0.01)
                
                # Apply custom threshold
                y_pred_custom_nb = (y_proba_nb[:, 1] >= nb_threshold).astype(int)
                
                # Calculate metrics with custom threshold
                custom_accuracy = accuracy_score(y_test, y_pred_custom_nb)
                custom_precision = precision_score(y_test, y_pred_custom_nb, zero_division=0)
                custom_recall = recall_score(y_test, y_pred_custom_nb, zero_division=0)
                custom_f1 = f1_score(y_test, y_pred_custom_nb, zero_division=0)
                
                # Display metrics with custom threshold
                col1, col2 = st.columns(2)
                col1.metric("Accuracy (Custom Threshold)", f"{custom_accuracy:.4f}")
                col1.metric("Precision (Custom Threshold)", f"{custom_precision:.4f}")
                col2.metric("Recall (Custom Threshold)", f"{custom_recall:.4f}")
                col2.metric("F1 Score (Custom Threshold)", f"{custom_f1:.4f}")
                
                # Display confusion matrix with custom threshold
                cm_custom_nb = confusion_matrix(y_test, y_pred_custom_nb)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm_custom_nb, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Legitimate', 'Fraud'], 
                           yticklabels=['Legitimate', 'Fraud'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f'Naive Bayes Confusion Matrix (Threshold={nb_threshold:.2f})')
                st.pyplot(fig)
        

    with model_tabs[2]:
        st.markdown("## Ridge e Lasso Regression para Detecção de Fraudes")

        st.write("""
        ### Regressão Regularizada para Classificação de Fraudes

        Embora Ridge e Lasso são técnicas de regressão, elas podem ser aplicadas para problemas de classificação
        binária como detecção de fraudes. Neste exemplo, usaremos essas técnicas para prever a variável 'Class'
        (0: transação legítima, 1: transação fraudulenta).
        
        - **Ridge Regression**: Utiliza regularização L2, que penaliza a soma dos quadrados dos coeficientes.
        - **Lasso Regression**: Utiliza regularização L1, que penaliza a soma dos valores absolutos dos coeficientes e pode reduzir alguns coeficientes a zero.
        """)
        
        # Seleção de variáveis
        st.subheader("Configuração do Modelo")

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
        
        st.write(f"Features selecionadas (baseadas em correlação com {target_column}):")
        st.write(best_features)
        
        # Dividir dados
        X = df[best_features].values
        y = df[target_column].values
        
        # Normalizar dados
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir em treino e teste
        test_size = st.slider("Proporção para teste (%)", 10, 40, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)
        
        # Configuração dos modelos
        st.subheader("Parâmetros de Regularização")
        
        col1, col2 = st.columns(2)
        with col1:
            alpha_ridge = st.slider(
                "Alpha para Ridge (força da regularização L2):", 
                0.01, 10.0, 1.0, 0.01
            )
    
        with col2:
            alpha_lasso = st.slider(
                "Alpha para Lasso (força da regularização L1):", 
                0.001, 1.0, 0.01, 0.001
            )
    
        # Treinamento dos modelos
        with st.spinner("Treinando modelos..."):
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
                "Acurácia": accuracy,
                "Precisão": precision,
                "Recall": recall, 
                "F1-Score": f1,
                "MSE": mse
            }
        
        # Mostrar resultados
        st.subheader("Resultados dos Modelos")
        
        # Criar dataframe de resultados
        results_df = pd.DataFrame({
            model: metrics
            for model, metrics in results.items()
        }).T
        
        st.write(results_df)
        
        # Gráfico de barras para F1-Score (melhor métrica para dados desbalanceados)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results_df.index, results_df["F1-Score"], color=["blue", "green", "orange"])
        ax.set_ylabel('F1-Score')
        ax.set_title('Comparação de Modelos - F1-Score (maior é melhor)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Visualizar coeficientes
        st.subheader("Coeficientes dos Modelos")
        
        coef_df = pd.DataFrame({
            'Feature': best_features,
            'Linear Regression': lr.coef_,
            f'Ridge (α={alpha_ridge})': ridge.coef_,
            f'Lasso (α={alpha_lasso})': lasso.coef_
        })
        
        st.write(coef_df.set_index('Feature'))
        
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
        ax.set_xlabel('Features')
        ax.set_ylabel('Coeficientes')
        ax.set_title('Importância das Features para Detecção de Fraudes')
        ax.set_xticks(index)
        ax.set_xticklabels(best_features, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Matriz de confusão para o melhor modelo
        st.subheader("Matriz de Confusão")
        
        # Encontrar o melhor modelo com base no F1-Score
        best_model_name = results_df["F1-Score"].idxmax()
        best_model_pred = predictions[best_model_name]
        
        cm = confusion_matrix(y_test, best_model_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
        ax.set_xlabel('Previsto')
        ax.set_ylabel('Real')
        ax.set_title(f'Matriz de Confusão - {best_model_name}')
        st.pyplot(fig)
        
        # Explicação sobre Ridge e Lasso para classificação
        st.subheader("Interpretação")
        
        st.write("""
        ### Aplicação de Ridge e Lasso para Detecção de Fraudes:
        
        1. **Interpretação dos Coeficientes**:
           - Coeficientes positivos: Indicam que valores maiores dessa feature aumentam a probabilidade de fraude
           - Coeficientes negativos: Indicam que valores maiores dessa feature diminuem a probabilidade de fraude
           - Coeficientes próximos a zero (especialmente em Lasso): Indicam features menos relevantes para a detecção
    
        2. **Comparação dos Modelos**:
           - **Regressão Linear**: Sem regularização, pode ser mais suscetível a overfitting, especialmente com muitas variáveis
           - **Ridge**: Reduz todos os coeficientes de forma proporcional, mantendo todas as features
           - **Lasso**: Tende a realizar seleção de features, eliminando algumas completamente (coeficientes = 0)
    
        3. **Por que usar regularização para fraudes?**
           - Dados de fraude geralmente têm muitas variáveis potencialmente correlacionadas
           - A regularização ajuda a evitar overfitting em dados de treinamento
           - Lasso pode identificar automaticamente as variáveis mais importantes para detecção
        """)
    
        # Adicionar thresholding interativo
        st.subheader("Ajuste de Limiar (Threshold)")
        
        st.write("""
        Em problemas de classificação desbalanceados como detecção de fraudes, 
        ajustar o limiar de decisão é crucial para equilibrar falsos positivos e falsos negativos.
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
        col1.metric("Acurácia", f"{custom_accuracy:.4f}")
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

    # Adicionar métodos de boosting
    st.subheader("🚀 Métodos de Boosting")

    st.write("""
    ### Métodos Ensemble: Boosting

    Algoritmos de Boosting constroem múltiplos modelos sequencialmente, onde cada modelo corrige 
    os erros de seus predecessores. Estes são particularmente eficazes para detecção de fraudes.

    - **AdaBoost**: Adaptive Boosting foca em instâncias classificadas incorretamente, aumentando seus pesos
    - **XGBoost**: eXtreme Gradient Boosting usa descida do gradiente para minimizar erros
    """)

    # Criar colunas para os dois modelos
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### AdaBoost")
        
        run_ada = st.checkbox("Treinar Modelo AdaBoost", value=False)
        if run_ada:
            from sklearn.ensemble import AdaBoostClassifier
            
            # Definir parâmetros
            n_estimators = st.slider("Número de Estimadores (AdaBoost)", 50, 300, 100)
            learning_rate = st.slider("Taxa de Aprendizado (AdaBoost)", 0.01, 2.0, 1.0, 0.01)
            
            with st.spinner("Treinando o modelo AdaBoost..."):
                # Iniciar contagem de tempo
                start = time.time()
                
                # Criar e treinar modelo
                ada_model = AdaBoostClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42
                )
                ada_model.fit(X_train_resampled, y_train_resampled)
                
                # Encerrar contagem de tempo
                duration = time.time() - start
                st.write(f"⏱️ Tempo de treinamento: {duration:.2f} segundos")
                
                # Fazer previsões
                y_pred_ada = ada_model.predict(X_test)
                
                # Calcular métricas
                ada_accuracy = accuracy_score(y_test, y_pred_ada)
                ada_precision = precision_score(y_test, y_pred_ada, zero_division=0)
                ada_recall = recall_score(y_test, y_pred_ada, zero_division=0)
                ada_f1 = f1_score(y_test, y_pred_ada, zero_division=0)
                
                # Exibir métricas
                st.metric("Acurácia", f"{ada_accuracy:.4f}")
                st.metric("Precisão", f"{ada_precision:.4f}")
                st.metric("Recall", f"{ada_recall:.4f}")
                st.metric("F1 Score", f"{ada_f1:.4f}")
                
                # Exibir matriz de confusão
                cm_ada = confusion_matrix(y_test, y_pred_ada)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
                ax.set_xlabel('Previsto')
                ax.set_ylabel('Real')
                ax.set_title('Matriz de Confusão AdaBoost')
                st.pyplot(fig)

    with col2:
        st.write("#### XGBoost")
        
        run_xgb = st.checkbox("Treinar Modelo XGBoost", value=False)
        if run_xgb:
            import xgboost as xgb
            
            # Definir parâmetros
            n_estimators_xgb = st.slider("Número de Estimadores (XGBoost)", 50, 300, 100)
            max_depth = st.slider("Profundidade Máxima (XGBoost)", 3, 10, 6)
            learning_rate_xgb = st.slider("Taxa de Aprendizado (XGBoost)", 0.01, 0.3, 0.1, 0.01)
            
            with st.spinner("Treinando o modelo XGBoost..."):
                # Iniciar contagem de tempo
                start = time.time()
                
                # Criar e treinar modelo
                xgb_model = xgb.XGBClassifier(
                    n_estimators=n_estimators_xgb,
                    max_depth=max_depth,
                    learning_rate=learning_rate_xgb,
                    random_state=42
                )
                xgb_model.fit(X_train_resampled, y_train_resampled)
                
                # Encerrar contagem de tempo
                duration = time.time() - start
                st.write(f"⏱️ Tempo de treinamento: {duration:.2f} segundos")
                
                # Fazer previsões
                y_pred_xgb = xgb_model.predict(X_test)
                
                # Calcular métricas
                xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
                xgb_precision = precision_score(y_test, y_pred_xgb, zero_division=0)
                xgb_recall = recall_score(y_test, y_pred_xgb, zero_division=0)
                xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
                
                # Exibir métricas
                st.metric("Acurácia", f"{xgb_accuracy:.4f}")
                st.metric("Precisão", f"{xgb_precision:.4f}")
                st.metric("Recall", f"{xgb_recall:.4f}")
                st.metric("F1 Score", f"{xgb_f1:.4f}")
                
                # Exibir matriz de confusão
                cm_xgb = confusion_matrix(y_test, y_pred_xgb)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
                ax.set_xlabel('Previsto')
                ax.set_ylabel('Real')
                ax.set_title('Matriz de Confusão XGBoost')
                st.pyplot(fig)
                
                # Importância das features para XGBoost
                fig, ax = plt.subplots(figsize=(8, 6))
                xgb.plot_importance(xgb_model, ax=ax, max_num_features=10)
                plt.title("Importância das Features - XGBoost")
                st.pyplot(fig)

    # Comparar modelos de boosting se ambos foram treinados
    if run_ada and run_xgb:
        st.subheader("Comparação dos Métodos de Boosting")
        
        # Criar dataframe de comparação
        boost_comparison = pd.DataFrame({
            'Modelo': ['AdaBoost', 'XGBoost'],
            'Acurácia': [ada_accuracy, xgb_accuracy],
            'Precisão': [ada_precision, xgb_precision],
            'Recall': [ada_recall, xgb_recall],
            'F1 Score': [ada_f1, xgb_f1]
        })
        
        st.write(boost_comparison)
        
        # Plotar comparação
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(2)
        width = 0.2
        metrics = ['Acurácia', 'Precisão', 'Recall', 'F1 Score']
        colors = ['blue', 'green', 'red', 'purple']
        
        for i, metric in enumerate(metrics):
            values = boost_comparison[metric].values
            ax.bar(x + i*width - 0.3, values, width, label=metric, color=colors[i])
        
        ax.set_xticks(x)
        ax.set_xticklabels(boost_comparison['Modelo'])
        ax.set_ylabel('Pontuação')
        ax.set_title('Comparação dos Métodos de Boosting')
        ax.legend()
        
        st.pyplot(fig)
        
        # Plotar tempos de treinamento
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(kernels, [training_times[k] for k in kernels], color='teal')
        ax.set_xlabel('Tipo de Kernel')
        ax.set_ylabel('Tempo de Treinamento (segundos)')
        ax.set_title('Tempo de Treinamento do SVM por Tipo de Kernel')
        for i, v in enumerate([training_times[k] for k in kernels]):
            ax.text(i, v + 0.1, f"{v:.2f}s", ha='center')
        st.pyplot(fig)
        
        st.write("""
        ### Escolhendo o K Ótimo
        
        A escolha do K é crucial para K-NN:
        
        - **Valores baixos de K** (ex: K=1): O modelo se torna muito sensível ao ruído
        - **Valores altos de K**: O modelo pode perder padrões importantes e subajustar
        
        Para detecção de fraudes:
        - Um K balanceado ajuda a evitar tanto a sobre-sensibilidade a dados ruidosos quanto a perda de sutis padrões de fraude
        - O K ótimo depende do tamanho do conjunto de dados e da distribuição das classes
        - Selecionamos o K que maximiza o F1 score para equilibrar precisão e recall
        """)

    with model_tabs[1]:
        # Adicionar seção para Naive Bayes
        st.subheader("📊 Naive Bayes Classifier")

        st.write("""
        ### Naive Bayes Classification

        Naive Bayes is a probabilistic classifier based on Bayes' theorem with an assumption of 
        independence between features. Despite this "naive" assumption, it works surprisingly 
        well in many real-world situations, including fraud detection.
        """)

        run_nb = st.checkbox("Train Naive Bayes Model", value=False)

        if run_nb:
            from sklearn.naive_bayes import GaussianNB
            
            with st.spinner("Training Naive Bayes model..."):
                # Start timing
                start = time.time()
                
                # Create and train model
                nb_model = GaussianNB()
                nb_model.fit(X_train_resampled, y_train_resampled)
                
                # End timing
                duration = time.time() - start
                st.write(f"⏱️ Training time: {duration:.2f} seconds")
                
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
                col1.metric("Precision", f"{nb_precision:.4f}")
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
                ax.set_title('Matriz de Confusão Naive Bayes')
                st.pyplot(fig)
                
                # Classification report
                st.write("### Naive Bayes Classification Report")
                st.text(classification_report(y_test, y_pred_nb))
                
                # Plot probability distribution
                st.write("### Fraud Probability Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(
                    data=pd.DataFrame({
                        'Fraud Probability': y_proba_nb[:, 1],
                        'Actual Class': y_test
                    }),
                    x='Fraud Probability',
                    hue='Actual Class',
                    bins=50,
                    ax=ax
                )
                plt.axvline(x=0.5, color='red', linestyle='--', label='Default threshold (0.5)')
                plt.legend()
                st.pyplot(fig)
                
                # Add custom threshold slider
                st.write("### Custom Threshold Adjustment")
                nb_threshold = st.slider("Decision Threshold for Naive Bayes", 0.0, 1.0, 0.5, 0.01)
                
                # Apply custom threshold
                y_pred_custom_nb = (y_proba_nb[:, 1] >= nb_threshold).astype(int)
                
                # Calculate metrics with custom threshold
                custom_accuracy = accuracy_score(y_test, y_pred_custom_nb)
                custom_precision = precision_score(y_test, y_pred_custom_nb, zero_division=0)
                custom_recall = recall_score(y_test, y_pred_custom_nb, zero_division=0)
                custom_f1 = f1_score(y_test, y_pred_custom_nb, zero_division=0)
                
                # Display metrics with custom threshold
                col1, col2 = st.columns(2)
                col1.metric("Accuracy (Custom Threshold)", f"{custom_accuracy:.4f}")
                col1.metric("Precision (Custom Threshold)", f"{custom_precision:.4f}")
                col2.metric("Recall (Custom Threshold)", f"{custom_recall:.4f}")
                col2.metric("F1 Score (Custom Threshold)", f"{custom_f1:.4f}")
                
                # Display confusion matrix with custom threshold
                cm_custom_nb = confusion_matrix(y_test, y_pred_custom_nb)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm_custom_nb, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Legítima', 'Fraude'], 
                           yticklabels=['Legítima', 'Fraude'])
                ax.set_xlabel('Previsto')
                ax.set_ylabel('Real')
                ax.set_title(f'Matriz de Confusão Naive Bayes (Limiar={nb_threshold:.2f})')
                st.pyplot(fig)
    
    with model_tabs[2]:
        st.markdown("## Ridge e Lasso Regression para Detecção de Fraudes")
        
        st.write("""
        ### Regressão Regularizada para Classificação de Fraudes
        
        Embora Ridge e Lasso são técnicas de regressão, elas podem ser aplicadas para problemas de classificação 
        binária como detecção de fraudes. Neste exemplo, usaremos essas técnicas para prever a variável 'Class'
        (0: transação legítima, 1: transação fraudulenta).
        
        - **Ridge Regression**: Utiliza regularização L2, que penaliza a soma dos quadrados dos coeficientes.
        - **Lasso Regression**: Utiliza regularização L1, que penaliza a soma dos valores absolutos dos coeficientes e pode reduzir alguns coeficientes a zero.
        """)
        
        # Seleção de variáveis
        st.subheader("Configuração do Modelo")

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
        
        st.write(f"Features selecionadas (baseadas em correlação com {target_column}):")
        st.write(best_features)
        
        # Dividir dados
        X = df[best_features].values
        y = df[target_column].values
        
        # Normalizar dados
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir em treino e teste
        test_size = st.slider("Proporção para teste (%)", 10, 40, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)
        
        # Configuração dos modelos
        st.subheader("Parâmetros de Regularização")
        
        col1, col2 = st.columns(2)
        with col1:
            alpha_ridge = st.slider(
                "Alpha para Ridge (força da regularização L2):", 
                0.01, 10.0, 1.0, 0.01
            )
    
        with col2:
            alpha_lasso = st.slider(
                "Alpha para Lasso (força da regularização L1):", 
                0.001, 1.0, 0.01, 0.001
            )
    
        # Treinamento dos modelos
        with st.spinner("Treinando modelos..."):
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
                "Acurácia": accuracy,
                "Precisão": precision,
                "Recall": recall, 
                "F1-Score": f1,
                "MSE": mse
            }
        
        # Mostrar resultados
        st.subheader("Resultados dos Modelos")
        
        # Criar dataframe de resultados
        results_df = pd.DataFrame({
            model: metrics
            for model, metrics in results.items()
        }).T
        
        st.write(results_df)
        
        # Gráfico de barras para F1-Score (melhor métrica para dados desbalanceados)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results_df.index, results_df["F1-Score"], color=["blue", "green", "orange"])
        ax.set_ylabel('F1-Score')
        ax.set_title('Comparação de Modelos - F1-Score (maior é melhor)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Visualizar coeficientes
        st.subheader("Coeficientes dos Modelos")
        
        coef_df = pd.DataFrame({
            'Feature': best_features,
            'Linear Regression': lr.coef_,
            f'Ridge (α={alpha_ridge})': ridge.coef_,
            f'Lasso (α={alpha_lasso})': lasso.coef_
        })
        
        st.write(coef_df.set_index('Feature'))
        
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
        ax.set_xlabel('Features')
        ax.set_ylabel('Coeficientes')
        ax.set_title('Importância das Features para Detecção de Fraudes')
        ax.set_xticks(index)
        ax.set_xticklabels(best_features, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Matriz de confusão para o melhor modelo
        st.subheader("Matriz de Confusão")
        
        # Encontrar o melhor modelo com base no F1-Score
        best_model_name = results_df["F1-Score"].idxmax()
        best_model_pred = predictions[best_model_name]
        
        cm = confusion_matrix(y_test, best_model_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
        ax.set_xlabel('Previsto')
        ax.set_ylabel('Real')
        ax.set_title(f'Matriz de Confusão - {best_model_name}')
        st.pyplot(fig)
        
        # Explicação sobre Ridge e Lasso para classificação
        st.subheader("Interpretação")
        
        st.write("""
        ### Aplicação de Ridge e Lasso para Detecção de Fraudes:
        
        1. **Interpretação dos Coeficientes**:
           - Coeficientes positivos: Indicam que valores maiores dessa feature aumentam a probabilidade de fraude
           - Coeficientes negativos: Indicam que valores maiores dessa feature diminuem a probabilidade de fraude
           - Coeficientes próximos a zero (especialmente em Lasso): Indicam features menos relevantes para a detecção
    
        2. **Comparação dos Modelos**:
           - **Regressão Linear**: Sem regularização, pode ser mais suscetível a overfitting, especialmente com muitas variáveis
           - **Ridge**: Reduz todos os coeficientes de forma proporcional, mantendo todas as features
           - **Lasso**: Tende a realizar seleção de features, eliminando algumas completamente (coeficientes = 0)
    
        3. **Por que usar regularização para fraudes?**
           - Dados de fraude geralmente têm muitas variáveis potencialmente correlacionadas
           - A regularização ajuda a evitar overfitting em dados de treinamento
           - Lasso pode identificar automaticamente as variáveis mais importantes para detecção
        """)
    
        # Adicionar thresholding interativo
        st.subheader("Ajuste de Limiar (Threshold)")
        
        st.write("""
        Em problemas de classificação desbalanceados como detecção de fraudes, 
        ajustar o limiar de decisão é crucial para equilibrar falsos positivos e falsos negativos.
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
        col1.metric("Acurácia", f"{custom_accuracy:.4f}")
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

    # Adicionar métodos de boosting
    st.subheader("🚀 Métodos de Boosting")

    st.write("""
    ### Métodos Ensemble: Boosting

    Algoritmos de Boosting constroem múltiplos modelos sequencialmente, onde cada modelo corrige 
    os erros de seus predecessores. Estes são particularmente eficazes para detecção de fraudes.

    - **AdaBoost**: Adaptive Boosting foca em instâncias classificadas incorretamente, aumentando seus pesos
    - **XGBoost**: eXtreme Gradient Boosting usa descida do gradiente para minimizar erros
    """)

    # Criar colunas para os dois modelos
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### AdaBoost")
        
        run_ada = st.checkbox("Treinar Modelo AdaBoost", value=False)
        if run_ada:
            from sklearn.ensemble import AdaBoostClassifier
            
            # Definir parâmetros
            n_estimators = st.slider("Número de Estimadores (AdaBoost)", 50, 300, 100)
            learning_rate = st.slider("Taxa de Aprendizado (AdaBoost)", 0.01, 2.0, 1.0, 0.01)
            
            with st.spinner("Treinando o modelo AdaBoost..."):
                # Iniciar contagem de tempo
                start = time.time()
                
                # Criar e treinar modelo
                ada_model = AdaBoostClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42
                )
                ada_model.fit(X_train_resampled, y_train_resampled)
                
                # Encerrar contagem de tempo
                duration = time.time() - start
                st.write(f"⏱️ Tempo de treinamento: {duration:.2f} segundos")
                
                # Fazer previsões
                y_pred_ada = ada_model.predict(X_test)
                
                # Calcular métricas
                ada_accuracy = accuracy_score(y_test, y_pred_ada)
                ada_precision = precision_score(y_test, y_pred_ada, zero_division=0)
                ada_recall = recall_score(y_test, y_pred_ada, zero_division=0)
                ada_f1 = f1_score(y_test, y_pred_ada, zero_division=0)
                
                # Exibir métricas
                st.metric("Acurácia", f"{ada_accuracy:.4f}")
                st.metric("Precisão", f"{ada_precision:.4f}")
                st.metric("Recall", f"{ada_recall:.4f}")
                st.metric("F1 Score", f"{ada_f1:.4f}")
                
                # Exibir matriz de confusão
                cm_ada = confusion_matrix(y_test, y_pred_ada)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
                ax.set_xlabel('Previsto')
                ax.set_ylabel('Real')
                ax.set_title('Matriz de Confusão AdaBoost')
                st.pyplot(fig)

    with col2:
        st.write("#### XGBoost")
        
        run_xgb = st.checkbox("Treinar Modelo XGBoost", value=False)
        if run_xgb:
            import xgboost as xgb
            
            # Definir parâmetros
            n_estimators_xgb = st.slider("Número de Estimadores (XGBoost)", 50, 300, 100)
            max_depth = st.slider("Profundidade Máxima (XGBoost)", 3, 10, 6)
            learning_rate_xgb = st.slider("Taxa de Aprendizado (XGBoost)", 0.01, 0.3, 0.1, 0.01)
            
            with st.spinner("Treinando o modelo XGBoost..."):
                # Iniciar contagem de tempo
                start = time.time()
                
                # Criar e treinar modelo
                xgb_model = xgb.XGBClassifier(
                    n_estimators=n_estimators_xgb,
                    max_depth=max_depth,
                    learning_rate=learning_rate_xgb,
                    random_state=42
                )
                xgb_model.fit(X_train_resampled, y_train_resampled)
                
                # Encerrar contagem de tempo
                duration = time.time() - start
                st.write(f"⏱️ Tempo de treinamento: {duration:.2f} segundos")
                
                # Fazer previsões
                y_pred_xgb = xgb_model.predict(X_test)
                
                # Calcular métricas
                xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
                xgb_precision = precision_score(y_test, y_pred_xgb, zero_division=0)
                xgb_recall = recall_score(y_test, y_pred_xgb, zero_division=0)
                xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
                
                # Exibir métricas
                st.metric("Acurácia", f"{xgb_accuracy:.4f}")
                st.metric("Precisão", f"{xgb_precision:.4f}")
                st.metric("Recall", f"{xgb_recall:.4f}")
                st.metric("F1 Score", f"{xgb_f1:.4f}")
                
                # Exibir matriz de confusão
                cm_xgb = confusion_matrix(y_test, y_pred_xgb)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
                ax.set_xlabel('Previsto')
                ax.set_ylabel('Real')
                ax.set_title('Matriz de Confusão XGBoost')
                st.pyplot(fig)
                
                # Importância das features para XGBoost
                fig, ax = plt.subplots(figsize=(8, 6))
                xgb.plot_importance(xgb_model, ax=ax, max_num_features=10)
                plt.title("Importância das Features - XGBoost")
                st.pyplot(fig)

    # Comparar modelos de boosting se ambos foram treinados
    if run_ada and run_xgb:
        st.subheader("Comparação dos Métodos de Boosting")
        
        # Criar dataframe de comparação
        boost_comparison = pd.DataFrame({
            'Modelo': ['AdaBoost', 'XGBoost'],
            'Acurácia': [ada_accuracy, xgb_accuracy],
            'Precisão': [ada_precision, xgb_precision],
            'Recall': [ada_recall, xgb_recall],
            'F1 Score': [ada_f1, xgb_f1]
        })
        
        st.write(boost_comparison)
        
        # Plotar comparação
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(2)
        width = 0.2
        metrics = ['Acurácia', 'Precisão', 'Recall', 'F1 Score']
        colors = ['blue', 'green', 'red', 'purple']
        
        for i, metric in enumerate(metrics):
            values = boost_comparison[metric].values
            ax.bar(x + i*width - 0.3, values, width, label=metric, color=colors[i])
        
        ax.set_xticks(x)
        ax.set_xticklabels(boost_comparison['Modelo'])
        ax.set_ylabel('Pontuação')
        ax.set_title('Comparação dos Métodos de Boosting')
        ax.legend()
        
        st.pyplot(fig)
        
        # Plotar tempos de treinamento
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(kernels, [training_times[k] for k in kernels], color='teal')
        ax.set_xlabel('Tipo de Kernel')
        ax.set_ylabel('Tempo de Treinamento (segundos)')
        ax.set_title('Tempo de Treinamento do SVM por Tipo de Kernel')
        for i, v in enumerate([training_times[k] for k in kernels]):
            ax.text(i, v + 0.1, f"{v:.2f}s", ha='center')
        st.pyplot(fig)
        
        st.write("""
        ### Escolhendo o K Ótimo
        
        A escolha do K é crucial para K-NN:
        
        - **Valores baixos de K** (ex: K=1): O modelo se torna muito sensível ao ruído
        - **Valores altos de K**: O modelo pode perder padrões importantes e subajustar
        
        Para detecção de fraudes:
        - Um K balanceado ajuda a evitar tanto a sobre-sensibilidade a dados ruidosos quanto a perda de sutis padrões de fraude
        - O K ótimo depende do tamanho do conjunto de dados e da distribuição das classes
        - Selecionamos o K que maximiza o F1 score para equilibrar precisão e recall
        """)

    with model_tabs[1]:
        # Adicionar seção para Naive Bayes
        st.subheader("📊 Naive Bayes Classifier")

        st.write("""
        ### Naive Bayes Classification

        Naive Bayes is a probabilistic classifier based on Bayes' theorem with an assumption of 
        independence between features. Despite this "naive" assumption, it works surprisingly 
        well in many real-world situations, including fraud detection.
        """)

        run_nb = st.checkbox("Train Naive Bayes Model", value=False)

        if run_nb:
            from sklearn.naive_bayes import GaussianNB
            
            with st.spinner("Training Naive Bayes model..."):
                # Start timing
                start = time.time()
                
                # Create and train model
                nb_model = GaussianNB()
                nb_model.fit(X_train_resampled, y_train_resampled)
                
                # End timing
                duration = time.time() - start
                st.write(f"⏱️ Training time: {duration:.2f} seconds")
                
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
                col1.metric("Precision", f"{nb_precision:.4f}")
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
                ax.set_title('Matriz de Confusão Naive Bayes')
                st.pyplot(fig)
                
                # Classification report
                st.write("### Naive Bayes Classification Report")
                st.text(classification_report(y_test, y_pred_nb))
                
                # Plot probability distribution
                st.write("### Fraud Probability Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(
                    data=pd.DataFrame({
                        'Fraud Probability': y_proba_nb[:, 1],
                        'Actual Class': y_test
                    }),
                    x='Fraud Probability',
                    hue='Actual Class',
                    bins=50,
                    ax=ax
                )
                plt.axvline(x=0.5, color='red', linestyle='--', label='Default threshold (0.5)')
                plt.legend()
                st.pyplot(fig)
                
                # Add custom threshold slider
                st.write("### Custom Threshold Adjustment")
                nb_threshold = st.slider("Decision Threshold for Naive Bayes", 0.0, 1.0, 0.5, 0.01)
                
                # Apply custom threshold
                y_pred_custom_nb = (y_proba_nb[:, 1] >= nb_threshold).astype(int)
                
                # Calculate metrics with custom threshold
                custom_accuracy = accuracy_score(y_test, y_pred_custom_nb)
                custom_precision = precision_score(y_test, y_pred_custom_nb, zero_division=0)
                custom_recall = recall_score(y_test, y_pred_custom_nb, zero_division=0)
                custom_f1 = f1_score(y_test, y_pred_custom_nb, zero_division=0)
                
                # Display metrics with custom threshold
                col1, col2 = st.columns(2)
                col1.metric("Accuracy (Custom Threshold)", f"{custom_accuracy:.4f}")
                col1.metric("Precision (Custom Threshold)", f"{custom_precision:.4f}")
                col2.metric("Recall (Custom Threshold)", f"{custom_recall:.4f}")
                col2.metric("F1 Score (Custom Threshold)", f"{custom_f1:.4f}")
                
                # Display confusion matrix with custom threshold
                cm_custom_nb = confusion_matrix(y_test, y_pred_custom_nb)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm_custom_nb, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Legítima', 'Fraude'], 
                           yticklabels=['Legítima', 'Fraude'])
                ax.set_xlabel('Previsto')
                ax.set_ylabel('Real')
                ax.set_title(f'Matriz de Confusão Naive Bayes (Limiar={nb_threshold:.2f})')
                st.pyplot(fig)
        
                # Add to model_tabs[1] after Naive Bayes section
        st.subheader("🧩 K-Nearest Neighbors")
        
        st.write("""
        ### K-Nearest Neighbors Classification
        
        K-NN is a non-parametric method that classifies new cases based on a majority vote of its k nearest neighbors.
        For fraud detection, it can identify transactions that are similar to known fraudulent patterns.
        """)
        
        run_knn = st.checkbox("Train K-NN Model and Find Optimal K", value=False)
        
        if run_knn:
            from sklearn.neighbors import KNeighborsClassifier
            
            with st.spinner("Finding optimal K and training K-NN model..."):
                # Find optimal K
                k_range = range(1, 31)
                k_scores = []
                k_times = []
                
                for k in k_range:
                    # Start timing
                    start = time.time()
                    
                    # Create and train model
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_train_resampled, y_train_resampled)
                    
                    # End timing
                    duration = time.time() - start
                    k_times.append(duration)
                    
                    # Get score
                    y_pred = knn.predict(X_test)
                    k_scores.append(f1_score(y_test, y_pred, zero_division=0))
                
                # Find best K
                best_k = k_range[np.argmax(k_scores)]
                
                # Train final model with best K
                best_knn = KNeighborsClassifier(n_neighbors=best_k)
                
                # Start timing
                start = time.time()
                best_knn.fit(X_train_resampled, y_train_resampled)
                duration = time.time() - start
                
                # Make predictions
                y_pred_knn = best_knn.predict(X_test)
                
                # Calculate metrics
                knn_accuracy = accuracy_score(y_test, y_pred_knn)
                knn_precision = precision_score(y_test, y_pred_knn, zero_division=0)
                knn_recall = recall_score(y_test, y_pred_knn, zero_division=0)
                knn_f1 = f1_score(y_test, y_pred_knn, zero_division=0)
            
            # Display optimal K
            st.write(f"### Optimal Number of Neighbors: K = {best_k}")
            st.write(f"⏱️ Training time: {duration:.2f} seconds")
            
            # Plot K vs F1 score
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k_range, k_scores, 'o-', markersize=8)
            ax.axvline(x=best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
            ax.set_xlabel('Number of Neighbors (K)')
            ax.set_ylabel('F1 Score')
            ax.set_title('F1 Score vs K Value')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            
            # Display metrics
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{knn_accuracy:.4f}")
            col1.metric("Precision", f"{knn_precision:.4f}")
            col2.metric("Recall", f"{knn_recall:.4f}")
            col2.metric("F1 Score", f"{knn_f1:.4f}")
            
            # Display confusion matrix
            cm_knn = confusion_matrix(y_test, y_pred_knn)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Legitimate', 'Fraud'], 
                       yticklabels=['Legitimate', 'Fraud'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'K-NN Confusion Matrix (K={best_k})')
            st.pyplot(fig)
            
            # Plot K vs Training Time
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k_range, k_times, 'o-', markersize=8, color='green')
            ax.set_xlabel('Number of Neighbors (K)')
            ax.set_ylabel('Training Time (seconds)')
            ax.set_title('Training Time vs K Value')
            ax.grid(True)
            st.pyplot(fig)
            
            st.write("""
            ### Choosing the Optimal K
            
            The choice of K is crucial for K-NN:
            
            - **Low K values** (e.g., K=1): Model becomes very sensitive to noise
            - **High K values**: Model may lose important patterns and underfit
            
            For fraud detection:
            - A balanced K helps avoid both over-sensitivity to noisy data and missing subtle fraud patterns
            - The optimal K depends on the dataset size and class distribution
            - We select the K that maximizes the F1 score to balance precision and recall
            """)
        
                # Add to model_tabs[1] after K-NN section
        st.subheader("🌳 Decision Trees")
        
        st.write("""
        ### Decision Tree Classification
        
        Decision Trees are intuitive models that make decisions by following branches of conditions.
        They're particularly useful for fraud detection because they can:
        - Capture non-linear relationships in data
        - Handle mixed types of features (numerical and categorical)
        - Provide transparent, explainable decisions
        """)
        
        run_dt = st.checkbox("Train Decision Tree Model", value=False)
        
        if run_dt:
            from sklearn.tree import DecisionTreeClassifier, plot_tree
            
            with st.spinner("Training Decision Tree model..."):
                # Parameters
                max_depth = st.slider("Maximum Tree Depth", 3, 15, 5)
                min_samples_split = st.slider("Minimum Samples to Split", 2, 50, 10)
                
                # Start timing
                start = time.time()
                
                # Create and train model
                dt_model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    class_weight='balanced',
                    random_state=42
                )
                dt_model.fit(X_train_resampled, y_train_resampled)
                
                # End timing
                duration = time.time() - start
                
                # Make predictions
                y_pred_dt = dt_model.predict(X_test)
                
                # Calculate metrics
                dt_accuracy = accuracy_score(y_test, y_pred_dt)
                dt_precision = precision_score(y_test, y_pred_dt, zero_division=0)
                dt_recall = recall_score(y_test, y_pred_dt, zero_division=0)
                dt_f1 = f1_score(y_test, y_pred_dt, zero_division=0)
            
            # Display training time
            st.write(f"⏱️ Training time: {duration:.2f} seconds")
            
            # Display metrics
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{dt_accuracy:.4f}")
            col1.metric("Precision", f"{dt_precision:.4f}")
            col2.metric("Recall", f"{dt_recall:.4f}")
            col2.metric("F1 Score", f"{dt_f1:.4f}")
            
            # Display confusion matrix
            cm_dt = confusion_matrix(y_test, y_pred_dt)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Legitimate', 'Fraud'], 
                       yticklabels=['Legitimate', 'Fraud'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Decision Tree Confusion Matrix')
            st.pyplot(fig)
            
            # Plot the tree
            st.write("### Visualization of the Decision Tree")
            
            # Get feature names
            if isinstance(X_train, pd.DataFrame):
                feature_names = X_train.columns.tolist()
            else:
                feature_names = features
            
            # Set figure size based on tree depth
            fig_height = max(6, max_depth * 2)
            fig_width = max(10, max_depth * 3)
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            plot_tree(dt_model, 
                      feature_names=feature_names,
                      class_names=['Legitimate', 'Fraud'],
                      filled=True, 
                      rounded=True,
                      ax=ax)
            st.pyplot(fig)
            
            # Feature importance
            st.write("### Feature Importance")
            importance = dt_model.feature_importances_
            
            # Create DataFrame for feature importance
            if isinstance(X_train, pd.DataFrame):
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
            else:
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
            
            st.write(feature_importance)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
            ax.set_title('Decision Tree Feature Importance')
            st.pyplot(fig)
            
            # Gini analysis
            st.write("### Gini Impurity Analysis")
            st.write("""
            Gini impurity measures how often a randomly chosen element would be incorrectly labeled 
            if it was randomly labeled according to the distribution of labels in the subset.
            
            - **Lower Gini value** = better split (more homogeneous nodes)
            - **Gini = 0** = all samples in a node are from the same class (perfect purity)
            - **Gini = 0.5** = equal distribution of classes (maximum impurity for binary classification)
            """)
            
            # Extract node information including Gini values
            n_nodes = dt_model.tree_.node_count
            children_left = dt_model.tree_.children_left
            children_right = dt_model.tree_.children_right
            feature = dt_model.tree_.feature
            threshold = dt_model.tree_.threshold
            impurity = dt_model.tree_.impurity
            
            # Create DataFrame for node information
            node_info = []
            for i in range(n_nodes):
                if children_left[i] != children_right[i]:  # not a leaf node
                    node_info.append({
                        'Node': i,
                        'Feature': feature_names[feature[i]] if feature[i] >= 0 else "N/A",
                        'Threshold': threshold[i],
                        'Gini Impurity': impurity[i],
                        'Left Child': children_left[i],
                        'Right Child': children_right[i]
                    })
            
            node_df = pd.DataFrame(node_info)
            if not node_df.empty:
                st.write("Top nodes by Gini impurity:")
                st.write(node_df.sort_values('Gini Impurity', ascending=True).head(10))
                
                # Plot Gini impurity for the first few levels
                if len(node_df) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(node_df['Node'].astype(str), node_df['Gini Impurity'], color='purple')
                    ax.set_xlabel('Node ID')
                    ax.set_ylabel('Gini Impurity')
                    ax.set_title('Gini Impurity by Node')
                    ax.set_xticks(range(min(20, len(node_df))))
                    plt.xticks(rotation=90)
                    st.pyplot(fig)

        # Add to model_tabs[1] after Decision Trees section
        st.subheader("🧠 Neural Networks")
        
        st.write("""
        ### Neural Network Models
        
        Neural Networks are versatile models inspired by the human brain's structure. They can:
        - Learn complex patterns in high-dimensional data
        - Adapt to changing fraud patterns
        - Process both numerical and categorical features
        
        We'll compare single-layer and multi-layer perceptrons for fraud detection.
        """)
        
        run_nn = st.checkbox("Train Neural Network Models", value=False)
        
        if run_nn:
            from sklearn.neural_network import MLPClassifier
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Single-Layer Perceptron")
                
                with st.spinner("Training Single-Layer Neural Network..."):
                    # Start timing
                    start = time.time()
                    
                    # Create and train model
                    slp_model = MLPClassifier(
                        hidden_layer_sizes=(100,),  # Single hidden layer
                        activation='relu',
                        solver='adam',
                        max_iter=300,
                        random_state=42
                    )
                    slp_model.fit(X_train_resampled, y_train_resampled)
                    
                    # End timing
                    slp_duration = time.time() - start
                    st.write(f"⏱️ Training time: {slp_duration:.2f} seconds")
                    
                    # Make predictions
                    y_pred_slp = slp_model.predict(X_test)
                    
                    # Calculate metrics
                    slp_accuracy = accuracy_score(y_test, y_pred_slp)
                    slp_precision = precision_score(y_test, y_pred_slp, zero_division=0)
                    slp_recall = recall_score(y_test, y_pred_slp, zero_division=0)
                    slp_f1 = f1_score(y_test, y_pred_slp, zero_division=0)
                    
                    # Display metrics
                    st.metric("Accuracy", f"{slp_accuracy:.4f}")
                    st.metric("Precision", f"{slp_precision:.4f}")
                    st.metric("Recall", f"{slp_recall:.4f}")
                    st.metric("F1 Score", f"{slp_f1:.4f}")
                    
                    # Display confusion matrix
                    cm_slp = confusion_matrix(y_test, y_pred_slp)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm_slp, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Legitimate', 'Fraud'], 
                               yticklabels=['Legitimate', 'Fraud'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Single-Layer NN Confusion Matrix')
                    st.pyplot(fig)
            
            with col2:
                st.write("#### Multi-Layer Perceptron")
                
                with st.spinner("Training Multi-Layer Neural Network..."):
                    # Start timing
                    start = time.time()
                    
                    # Create and train model
                    mlp_model = MLPClassifier(
                        hidden_layer_sizes=(100, 50),  # Multiple hidden layers
                        activation='relu',
                        solver='adam',
                        max_iter=300,
                        random_state=42
                    )
                    mlp_model.fit(X_train_resampled, y_train_resampled)
                    
                    # End timing
                    mlp_duration = time.time() - start
                    st.write(f"⏱️ Training time: {mlp_duration:.2f} seconds")
                    
                    # Make predictions
                    y_pred_mlp = mlp_model.predict(X_test)
                    
                    # Calculate metrics
                    mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
                    mlp_precision = precision_score(y_test, y_pred_mlp, zero_division=0)
                    mlp_recall = recall_score(y_test, y_pred_mlp, zero_division=0)
                    mlp_f1 = f1_score(y_test, y_pred_mlp, zero_division=0)
                    
                    # Display metrics
                    st.metric("Accuracy", f"{mlp_accuracy:.4f}")
                    st.metric("Precision", f"{mlp_precision:.4f}")
                    st.metric("Recall", f"{mlp_recall:.4f}")
                    st.metric("F1 Score", f"{mlp_f1:.4f}")
                    
                    # Display confusion matrix
                    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Legitimate', 'Fraud'], 
                               yticklabels=['Legitimate', 'Fraud'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Multi-Layer NN Confusion Matrix')
                    st.pyplot(fig)
            
            # Compare models
            st.write("### Neural Network Architecture Comparison")
            
            # Create comparison dataframe
            nn_comparison = pd.DataFrame({
                'Model': ['Single-Layer NN', 'Multi-Layer NN'],
                'Accuracy': [slp_accuracy, mlp_accuracy],
                'Precision': [slp_precision, mlp_precision],
                'Recall': [slp_recall, mlp_recall],
                'F1 Score': [slp_f1, mlp_f1],
                'Training Time': [slp_duration, mlp_duration]
            })
            
            st.write(nn_comparison)
            
            # Plot loss curves if available
            if hasattr(slp_model, 'loss_curve_') and hasattr(mlp_model, 'loss_curve_'):
                st.write("### Training Loss Curves")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(slp_model.loss_curve_, label='Single-Layer')
                ax.plot(mlp_model.loss_curve_, label='Multi-Layer')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Loss')
                ax.set_title('Neural Network Training Loss')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            # NN Architecture visualization
            st.write("### Neural Network Architecture")
            
            st.write("""
            **Single-Layer Network:**  
            Input Layer (features) → Hidden Layer (100 neurons) → Output Layer (1 neuron)
            
            **Multi-Layer Network:**  
            Input Layer (features) → Hidden Layer 1 (100 neurons) → Hidden Layer 2 (50 neurons) → Output Layer (1 neuron)
            """)
            
            # Simple architecture visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Single layer visualization (simplified)
            n_input = X_train.shape[1]
            n_hidden1 = 100
            n_output = 1
            
            # Draw nodes
            for i, layer_size in enumerate([n_input, n_hidden1, n_output]):
                x = i
                for j in range(min(layer_size, 10)):  # Limit to 10 nodes per layer for visualization
                    y = j * 0.1
                    if j == 9 and layer_size > 10:
                        ax1.scatter(x, y, s=100, c='blue', alpha=0.8)
                        ax1.text(x, y+0.02, "...", ha='center')
                    else:
                        ax1.scatter(x, y, s=100, c='blue', alpha=0.8)
            
            ax1.set_xticks([0, 1, 2])
            ax1.set_xticklabels(['Input', 'Hidden', 'Output'])
            ax1.set_yticks([])
            ax1.set_title("Single-Layer Network")
            
            # Multi-layer visualization (simplified)
            n_hidden2 = 50
            
            # Draw nodes
            for i, layer_size in enumerate([n_input, n_hidden1, n_hidden2, n_output]):
                x = i
                for j in range(min(layer_size, 10)):  # Limit to 10 nodes per layer for visualization
                    y = j * 0.1
                    if j == 9 and layer_size > 10:
                        ax2.scatter(x, y, s=100, c='red', alpha=0.8)
                        ax2.text(x, y+0.02, "...", ha='center')
                    else:
                        ax2.scatter(x, y, s=100, c='red', alpha=0.8)
            
            ax2.set_xticks([0, 1, 2, 3])
            ax2.set_xticklabels(['Input', 'Hidden 1', 'Hidden 2', 'Output'])
            ax2.set_yticks([])
            ax2.set_title("Multi-Layer Network")
            
            st.pyplot(fig)
        
                # Add to model_tabs[1] after all models
        st.subheader("🔄 Cross Validation")
        
        st.write("""
        ### K-Fold Cross Validation
        
        Cross-validation helps ensure our model performs consistently across different subsets of data.
        This is particularly important for fraud detection, where we need to be confident our model 
        will generalize well to new, unseen transactions.
        """)
        
        run_cv = st.checkbox("Run Cross Validation", value=False)
        
        if run_cv:
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            
            # Select model for cross validation
            cv_model_type = st.selectbox(
                "Select model for cross-validation:",
                ["Random Forest", "Logistic Regression", "XGBoost", "SVM", "K-NN", "Decision Tree", "Neural Network"]
            )
            
            # Number of folds
            n_folds = st.slider("Number of folds", 3, 10, 5)
            
            with st.spinner(f"Running {n_folds}-fold cross-validation for {cv_model_type}..."):
                # Create model based on selection
                if cv_model_type == "Random Forest":
                    from sklearn.ensemble import RandomForestClassifier
                    cv_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                elif cv_model_type == "Logistic Regression":
                    from sklearn.linear_model import LogisticRegression
                    cv_model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=2000, random_state=42)
                elif cv_model_type == "XGBoost":
                    import xgboost as xgb
                    cv_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
                elif cv_model_type == "SVM":
                    from sklearn.svm import SVC
                    cv_model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
                elif cv_model_type == "K-NN":
                    from sklearn.neighbors import KNeighborsClassifier
                    cv_model = KNeighborsClassifier(n_neighbors=5)
                elif cv_model_type == "Decision Tree":
                    from sklearn.tree import DecisionTreeClassifier
                    cv_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
                else:  # Neural Network
                    from sklearn.neural_network import MLPClassifier
                    cv_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
                
                # Define metrics to evaluate
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                cv_results = {}
                cv_times = []
                
                # Use stratified k-fold to preserve class distribution
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                
                # Run cross-validation for each metric
                for metric in metrics:
                    start = time.time()
                    scores = cross_val_score(cv_model, X, y, cv=skf, scoring=metric)
                    duration = time.time() - start
                    cv_times.append(duration / n_folds)  # Average time per fold
                    cv_results[metric] = scores
            
            # Display cross-validation results
            st.write(f"### Cross-Validation Results for {cv_model_type}")
            
            # Create summary table
            summary = pd.DataFrame({
                'Metric': metrics,
                'Mean': [cv_results[m].mean() for m in metrics],
                'Std Dev': [cv_results[m].std() for m in metrics],
                'Min': [cv_results[m].min() for m in metrics],
                'Max': [cv_results[m].max() for m in metrics]
            })
            
            st.write(summary)
            
            # Plot cross-validation results
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Set width of bars
            width = 0.2
            x = np.arange(n_folds)
            
            # Plot bars for each metric
            for i, metric in enumerate(metrics):
                ax.bar(x + i*width - 0.3, cv_results[metric], width, label=metric)
            
            # Add labels and legend
            ax.set_xlabel('Fold')
            ax.set_ylabel('Score')
            ax.set_title(f'Cross-Validation Scores for {cv_model_type}')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
            ax.legend()
            
            st.pyplot(fig)
            
            # Display average time per fold
            st.write("### Cross-Validation Time")
            avg_time = sum(cv_times) / len(cv_times)
            st.write(f"Average time per fold: {avg_time:.2f} seconds")
            st.write(f"Total cross-validation time: {sum(cv_times):.2f} seconds")
            
            # Plot time per metric
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(metrics, cv_times)
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Time per Metric in Cross-Validation')
            st.pyplot(fig)
            
            # Explanation
            st.write("""
            ### Interpreting Cross-Validation Results
            
            - **Mean**: Average performance across all folds
            - **Std Dev**: Variation in performance between folds - lower is better
            - **Min/Max**: Range of performance across folds
            
            A good model should have:
            1. High mean performance (especially F1 score for fraud detection)
            2. Low standard deviation (consistent performance)
            3. Reasonable training time (for real-world application)
            """)
        
                # Add to model_tabs[1] after cross validation
        st.subheader("🔄 Principal Component Analysis (PCA)")
        
        st.write("""
        ### Dimensionality Reduction with PCA
        
        Principal Component Analysis (PCA) is a technique for reducing the dimensionality of data
        while preserving as much variance as possible. For fraud detection:
        
        - It can help remove noise and redundancy in features
        - Speeds up model training and prediction
        - Can improve model performance by focusing on the most informative directions in the data
        """)
        
        run_pca = st.checkbox("Run PCA Analysis", value=False)
        
        if run_pca:
            from sklearn.decomposition import PCA
            
            # Number of components
            n_components = st.slider("Number of PCA components", 2, min(20, X.shape[1]), 5)
            
            with st.spinner(f"Running PCA with {n_components} components..."):
                # Start timing
                start = time.time()
                
                # Create and fit PCA
                pca = PCA(n_components=n_components, svd_solver='full')
                X_pca = pca.fit_transform(X)
                
                # End timing
                duration = time.time() - start
                
                # Split PCA-transformed data
                X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
                    X_pca, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Apply SMOTE to balance classes
                smote = SMOTE(random_state=42)
                X_train_pca_resampled, y_train_pca_resampled = smote.fit_resample(X_train_pca, y_train_pca)
            
            # Display PCA results
            st.write(f"PCA completed in {duration:.2f} seconds")
            st.write(f"Reduced feature space from {X.shape[1]} to {n_components} dimensions")
            
            # Explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            st.write("### Explained Variance")
            
            # Plot explained variance
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Individual variance
            ax[0].bar(range(1, n_components + 1), explained_variance, alpha=0.7, align='center')
            ax[0].set_xlabel('Principal Component')
            ax[0].set_ylabel('Explained Variance Ratio')
            ax[0].set_title('Variance Explained by Each Component')
            
            # Cumulative variance
            ax[1].plot(range(1, n_components + 1), cumulative_variance, 'o-', linewidth=2)
            ax[1].set_xlabel('Number of Components')
            ax[1].set_ylabel('Cumulative Explained Variance')
            ax[1].set_title(f'Cumulative Variance ({cumulative_variance[-1]:.2%} Total)')
            ax[1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
            ax[1].grid(True)
            ax[1].legend()
            
            st.pyplot(fig)
            
            # Compare model performance with and without PCA
            st.write("### Model Performance Comparison: PCA vs Raw Data")
            
            # Select model for comparison
            pca_model_type = st.selectbox(
                "Select model to compare with PCA:",
                ["Random Forest", "Logistic Regression", "XGBoost", "SVM"]
            )
            
            with st.spinner(f"Training {pca_model_type} on raw data and PCA data..."):
                # Create model based on selection
                if pca_model_type == "Random Forest":
                    from sklearn.ensemble import RandomForestClassifier
                    model_raw = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                    model_pca = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                elif pca_model_type == "Logistic Regression":
                    from sklearn.linear_model import LogisticRegression
                    model_raw = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=2000, random_state=42)
                    model_pca = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=2000, random_state=42)
                elif pca_model_type == "XGBoost":
                    import xgboost as xgb
                    model_raw = xgb.XGBClassifier(n_estimators=100, random_state=42)
                    model_pca = xgb.XGBClassifier(n_estimators=100, random_state=42)
                else:  # SVM
                    from sklearn.svm import SVC
                    model_raw = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
                    model_pca = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
                
                # Train and time model on raw data
                start_raw = time.time()
                model_raw.fit(X_train_resampled, y_train_resampled)
                raw_duration = time.time() - start_raw
                
                # Make predictions on raw data
                y_pred_raw = model_raw.predict(X_test)
                
                # Calculate metrics for raw data
                raw_accuracy = accuracy_score(y_test, y_pred_raw)
                raw_precision = precision_score(y_test, y_pred_raw, zero_division=0)
                raw_recall = recall_score(y_test, y_pred_raw, zero_division=0)
                raw_f1 = f1_score(y_test, y_pred_raw, zero_division=0)
                
                # Train and time model on PCA data
                start_pca = time.time()
                model_pca.fit(X_train_pca_resampled, y_train_pca_resampled)
                pca_duration = time.time() - start_pca
                
                # Make predictions on PCA data
                y_pred_pca = model_pca.predict(X_test_pca)
                
                # Calculate metrics for PCA data
                pca_accuracy = accuracy_score(y_test_pca, y_pred_pca)
                pca_precision = precision_score(y_test_pca, y_pred_pca, zero_division=0)
                pca_recall = recall_score(y_test_pca, y_pred_pca, zero_division=0)
                pca_f1 = f1_score(y_test_pca, y_pred_pca, zero_division=0)
            
            # Display comparison
            comparison = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time (s)'],
                'Raw Data': [raw_accuracy, raw_precision, raw_recall, raw_f1, raw_duration],
                'PCA Data': [pca_accuracy, pca_precision, pca_recall, pca_f1, pca_duration]
            })
            
            st.write(comparison.set_index('Metric'))
            
            # Calculate improvement percentages
            time_improvement = (raw_duration - pca_duration) / raw_duration * 100
            f1_change = (pca_f1 - raw_f1) / raw_f1 * 100
            
            # Display improvement summary
            st.write(f"""
            ### PCA Impact Summary
            
            - **Training Time**: {'Reduced' if time_improvement > 0 else 'Increased'} by {abs(time_improvement):.2f}%
            - **F1 Score**: {'Improved' if f1_change > 0 else 'Reduced'} by {abs(f1_change):.2f}%
            """)
            
            # Plot comparison
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            raw_values = [raw_accuracy, raw_precision, raw_recall, raw_f1]
            pca_values = [pca_accuracy, pca_precision, pca_recall, pca_f1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(metrics))
            width = 0.35
            
            ax.bar(x - width/2, raw_values, width, label='Raw Data')
            ax.bar(x + width/2, pca_values, width, label='PCA Data')
            
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.set_ylim(0, 1)
            ax.set_title(f'Performance Comparison: Raw vs. PCA Data ({pca_model_type})')
            ax.set_ylabel('Score')
            
            st.pyplot(fig)
            
            # Plot training time comparison
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(['Raw Data', 'PCA Data'], [raw_duration, pca_duration])
            ax.set_ylabel('Training Time (seconds)')
            ax.set_title(f'Training Time Comparison ({pca_model_type})')
            
            for i, v in enumerate([raw_duration, pca_duration]):
                ax.text(i, v + 0.1, f"{v:.2f}s", ha='center')
            
            st.pyplot(fig)
            
            # Visualization of data in PCA space
            if n_components >= 2:
                st.write("### Visualization of Data in PCA Space")
                
                # Plot in 2D space
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6, s=30)
                ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)')
                ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)')
                ax.set_title('Fraud vs. Legitimate Transactions in PCA Space')
                
                # Add legend
                legend_labels = {0: 'Legitimate', 1: 'Fraud'}
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=scatter.cmap(scatter.norm(c)), 
                                             markersize=10, label=legend_labels[c]) 
                                 for c in [0, 1]]
                ax.legend(handles=legend_elements)
                
                # Add grid
                ax.grid(alpha=0.3)
                
                st.pyplot(fig)
                
                # 3D plot if we have at least 3 components
                if n_components >= 3:
                    st.write("### 3D Visualization of PCA Data")
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(
                        X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                        c=y, cmap='coolwarm', alpha=0.6, s=30
                    )
                    
                    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
                    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
                    ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
                    ax.set_title('3D PCA Projection of Credit Card Transactions')
                    
                    # Add legend
                    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=scatter.cmap(scatter.norm(c)), 
                                                markersize=10, label=legend_labels[c]) 
                                    for c in [0, 1]]
                    ax.legend(handles=legend_elements)
                    
                    st.pyplot(fig)

                    # Add to model_tabs[1] after PCA section
            st.subheader("🌳 Hierarchical Clustering")
            
            st.write("""
            ### Hierarchical Clustering Analysis
            
            Hierarchical clustering builds a tree-like structure of nested clusters by repeatedly merging or 
            splitting groups. This helps identify natural groupings in transaction data and potential anomalies.
            
            Key benefits for fraud detection:
            - Reveals the natural structure of transaction data
            - Identifies outliers that may represent fraudulent activity
            - Shows relationships between different transaction types
            - Doesn't require pre-specifying the number of clusters
            """)
            
            run_hc = st.checkbox("Run Hierarchical Clustering", value=False)
            
            if run_hc:
                from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
                from sklearn.preprocessing import StandardScaler
                
                # Select features for clustering
                st.write("### Select Features for Clustering")
                feature_options = list(df.select_dtypes(include=['number']).columns)
                selected_features = st.multiselect(
                    "Select features for clustering (2-3 recommended):",
                    options=feature_options,
                    default=feature_options[:3] if len(feature_options) >= 3 else feature_options
                )
                
                if len(selected_features) < 2:
                    st.warning("Please select at least 2 features for clustering.")
                else:
                    # Sample size control
                    sample_size = st.slider(
                        "Number of transactions to sample (larger samples take longer)",
                        min_value=100,
                        max_value=2000,
                        value=500,
                        step=100
                    )
                    
                    # Get a stratified sample
                    fraud_sample = df[df['Class'] == 1].sample(
                        min(int(sample_size * 0.3), len(df[df['Class'] == 1])), 
                        random_state=42
                    )
                    normal_sample = df[df['Class'] == 0].sample(
                        sample_size - len(fraud_sample), 
                        random_state=42
                    )
                    sample_df = pd.concat([fraud_sample, normal_sample])
                    
                    # Extract features and scale them
                    X_cluster = sample_df[selected_features].values
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_cluster)
                    
                    # Method selection
                    linkage_method = st.selectbox(
                        "Select linkage method:",
                        options=["ward", "complete", "average", "single"],
                        index=0,
                        help="Ward minimizes variance within clusters. Complete uses max distance between points. Average uses mean distance. Single uses min distance."
                    )
                    
                    with st.spinner(f"Performing hierarchical clustering with {linkage_method} linkage method..."):
                        # Compute the linkage matrix
                        Z = linkage(X_scaled, method=linkage_method)
                        
                        # Create the dendrogram
                        fig, ax = plt.subplots(figsize=(12, 8))
                        dendrogram(
                            Z,
                            leaf_rotation=90.,
                            leaf_font_size=8.,
                            ax=ax
                        )
                        ax.set_title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage)')
                        ax.set_xlabel('Sample index')
                        ax.set_ylabel('Distance')
                        st.pyplot(fig)
                    
                    # Determine optimal number of clusters
                    st.write("### Determining Optimal Number of Clusters")
                    
                    # Calculate distances at which clusters merge
                    last_merge = Z[-10:, 2]
                    merge_diffs = np.diff(last_merge)
                    
                    # Plot the distances
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(range(2, len(merge_diffs) + 2), merge_diffs, 'o-')
                    ax.set_xticks(range(2, len(merge_diffs) + 2))
                    ax.set_xlabel('Number of clusters')
                    ax.set_ylabel('Distance increase')
                    ax.set_title('Increase in Distance as Clusters Merge')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Suggest optimal number based on elbow method
                    largest_increase_idx = np.argmax(merge_diffs) + 2
                    st.write(f"Based on the elbow method, **{largest_increase_idx}** clusters is suggested as optimal.")
                    
                    # Let user select number of clusters
                    n_clusters = st.slider(
                        "Select number of clusters to form:",
                        min_value=2,
                        max_value=10,
                        value=largest_increase_idx
                    )
                    
                    # Form clusters
                    with st.spinner(f"Forming {n_clusters} clusters..."):
                        # Create a simplified dendrogram showing the selected clusters
                        fig, ax = plt.subplots(figsize=(12, 8))
                        dendrogram(
                            Z,
                            truncate_mode='lastp',
                            p=n_clusters,
                            leaf_rotation=90.,
                            leaf_font_size=10.,
                            show_contracted=True,
                            ax=ax
                        )
                        ax.set_title(f'Simplified Dendrogram with {n_clusters} Clusters')
                        ax.axhline(y=Z[-(n_clusters-1), 2], color='red', linestyle='--')
                        st.pyplot(fig)
                        
                        # Assign cluster labels
                        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
                        
                        # Add clusters to the dataframe
                        sample_df['Cluster'] = cluster_labels
                        
                        # Show cluster statistics
                        st.write("### Cluster Analysis")
                        
                        # Create cluster stats table
                        cluster_stats = pd.DataFrame()
                        for i in range(1, n_clusters + 1):
                            cluster_data = sample_df[sample_df['Cluster'] == i]
                            
                            # Calculate fraud percentage
                            fraud_percent = cluster_data['Class'].mean() * 100
                            
                            # Store statistics
                            cluster_stats = pd.concat([cluster_stats, pd.DataFrame({
                                'Cluster': [i],
                                'Size': [len(cluster_data)],
                                'Fraud %': [f"{fraud_percent:.2f}%"],
                                'Mean Amount': [f"${cluster_data['Amount'].mean():.2f}"],
                                'Max Amount': [f"${cluster_data['Amount'].max():.2f}"],
                            })])
                        
                        st.write(cluster_stats.set_index('Cluster'))
                        
                        # Highlight potentially fraudulent clusters
                        fraud_clusters = cluster_stats[cluster_stats['Fraud %'].astype(str).str.replace('%', '').astype(float) > 10]
                        if not fraud_clusters.empty:
                            st.write("### 🚨 Potential Fraud Clusters")
                            st.write(fraud_clusters.set_index('Cluster'))
                            st.write("These clusters have higher than 10% fraud transactions and may warrant further investigation.")
                        
                        # Visualize clusters (if we selected 2 or 3 features)
                        st.write("### Cluster Visualization")
                        if len(selected_features) >= 3:
                            # 3D visualization
                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111, projection='3d')
                            
                            # Color by cluster
                            scatter = ax.scatter(
                                X_scaled[:, 0], 
                                X_scaled[:, 1], 
                                X_scaled[:, 2],
                                c=cluster_labels, 
                                cmap='viridis', 
                                s=50, 
                                alpha=0.8
                            )
                            
                            # Add fraud points with different marker
                            fraud_indices = sample_df['Class'] == 1
                            if fraud_indices.sum() > 0:
                                ax.scatter(
                                    X_scaled[fraud_indices, 0],
                                    X_scaled[fraud_indices, 1],
                                    X_scaled[fraud_indices, 2],
                                    c='red',
                                    marker='x',
                                    s=100,
                                    label='Fraud'
                                )
                            
                            ax.set_xlabel(selected_features[0])
                            ax.set_ylabel(selected_features[1])
                            ax.set_zlabel(selected_features[2])
                            ax.set_title(f'3D Cluster Visualization ({n_clusters} clusters)')
                            
                            # Add legend
                            ax.legend()
                            st.pyplot(fig)
                            
                        else:
                            # 2D visualization
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            # Color by cluster
                            scatter = ax.scatter(
                                X_scaled[:, 0],
                                X_scaled[:, 1],
                                c=cluster_labels,
                                cmap='viridis',
                                s=50,
                                alpha=0.8
                            )
                            
                            # Add fraud points with different marker
                            fraud_indices = sample_df['Class'] == 1
                            if fraud_indices.sum() > 0:
                                ax.scatter(
                                    X_scaled[fraud_indices, 0],
                                    X_scaled[fraud_indices, 1],
                                    c='red',
                                    marker='x',
                                    s=100,
                                    label='Fraud'
                                )
                            
                            ax.set_xlabel(selected_features[0])
                            ax.set_ylabel(selected_features[1])
                            ax.set_title(f'2D Cluster Visualization ({n_clusters} clusters)')
                            
                            # Add legend
                            ax.legend()
                            st.pyplot(fig)
                        
                        # Fraud distribution across clusters
                        st.write("### Fraud Distribution Across Clusters")
                        
                        # Create a dataframe of cluster vs fraud
                        fraud_distribution = pd.crosstab(
                            sample_df['Cluster'], 
                            sample_df['Class'],
                            rownames=['Cluster'], 
                            colnames=['Class'],
                            normalize='index'
                        ) * 100
                        
                        # Rename columns
                        fraud_distribution.columns = ['Legitimate %', 'Fraud %']
                        
                        # Display the table
                        st.write(fraud_distribution.round(2))
                        
                        # Plot a stacked bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        fraud_distribution.plot(
                            kind='bar',
                            stacked=True,
                            ax=ax,
                            color=['lightblue', 'salmon']
                        )
                        ax.set_ylabel('Percentage')
                        ax.set_title('Percentage of Fraud vs Legitimate Transactions by Cluster')
                        ax.legend(loc='upper right')
                        
                        # Add text labels
                        for i, p in enumerate(ax.patches):
                            width = p.get_width()
                            height = p.get_height()
                            x, y = p.get_xy() 
                            if height > 5:  # Only show label if segment is big enough
                                ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), 
                                            ha='center', va='center')
                                            
                        st.pyplot(fig)
            
                        # Feature importance by cluster
                        st.write("### Feature Distributions by Cluster")
                        
                        # Select a feature to analyze
                        feature_to_analyze = st.selectbox(
                            "Select a feature to analyze across clusters:",
                            selected_features
                        )
                        
                        # Create a boxplot of the feature values by cluster
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(
                            x='Cluster',
                            y=feature_to_analyze,
                            data=sample_df,
                            ax=ax,
                            palette='viridis'
                        )
                        ax.set_title(f'Distribution of {feature_to_analyze} by Cluster')
                        st.pyplot(fig)
                    
                        # Final insights
                        st.write("### Key Insights from Hierarchical Clustering")
                        
                        # Identify the cluster with highest fraud rate
                        highest_fraud_cluster = fraud_distribution['Fraud %'].idxmax()
                        highest_fraud_rate = fraud_distribution.loc[highest_fraud_cluster, 'Fraud %']
                        
                        st.write(f"""
                        - **Cluster {highest_fraud_cluster}** has the highest fraud rate at **{highest_fraud_rate:.2f}%**
                        - We detected **{n_clusters}** distinct transaction patterns
                        - Clusters show how transactions naturally group based on the selected features
                        - Transactions in the same cluster share similar characteristics
                        - Outliers or small clusters with high fraud rates may indicate specialized fraud patterns
                        """)
            else:
                st.info("Select the checkbox to run hierarchical clustering analysis.")
            
            # Add explanation about hierarchical clustering for fraud detection
            st.write("""
            ### Using Hierarchical Clustering for Fraud Detection
            
            Hierarchical clustering can be particularly valuable for fraud detection because:
            
            1. **Natural Transaction Segmentation**: It reveals natural groupings in transaction data without requiring a pre-defined number of clusters
            
            2. **Anomaly Detection**: Isolated branches or small clusters in the dendrogram often represent anomalous behavior
            
            3. **Investigation Prioritization**: Clusters with high fraud percentages can be prioritized for investigation
            
            4. **Data Exploration**: The dendrogram provides a visual map of how different transaction types relate to each other
            
            5. **Adaptive Fraud Detection**: By understanding the hierarchical structure of transactions, fraud detection systems can adapt to different transaction profiles
            
            The dendrogram above shows how transactions are progressively merged into clusters based on their similarity, with the vertical axis representing the distance (dissimilarity) between clusters.
            """)      
        
    
    with model_tabs[2]:
        st.markdown("## Ridge e Lasso Regression para Detecção de Fraudes")
        
        st.write("""
        ### Regressão Regularizada para Classificação de Fraudes
        
        Embora Ridge e Lasso são técnicas de regressão, elas podem ser aplicadas para problemas de classificação 
        binária como detecção de fraudes. Neste exemplo, usaremos essas técnicas para prever a variável 'Class'
        (0: transação legítima, 1: transação fraudulenta).
        
        - **Ridge Regression**: Utiliza regularização L2, que penaliza a soma dos quadrados dos coeficientes.
        - **Lasso Regression**: Utiliza regularização L1, que penaliza a soma dos valores absolutos dos coeficientes e pode reduzir alguns coeficientes a zero.
        """)
        
        # Seleção de variáveis
        st.subheader("Configuração do Modelo")

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
        
        st.write(f"Features selecionadas (baseadas em correlação com {target_column}):")
        st.write(best_features)
        
        # Dividir dados
        X = df[best_features].values
        y = df[target_column].values
        
        # Normalizar dados
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir em treino e teste
        test_size = st.slider("Proporção para teste (%)", 10, 40, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)
        
        # Configuração dos modelos
        st.subheader("Parâmetros de Regularização")
        
        col1, col2 = st.columns(2)
        with col1:
            alpha_ridge = st.slider(
                "Alpha para Ridge (força da regularização L2):", 
                0.01, 10.0, 1.0, 0.01
            )
    
        with col2:
            alpha_lasso = st.slider(
                "Alpha para Lasso (força da regularização L1):", 
                0.001, 1.0, 0.01, 0.001
            )
    
        # Treinamento dos modelos
        with st.spinner("Treinando modelos..."):
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
                "Acurácia": accuracy,
                "Precisão": precision,
                "Recall": recall, 
                "F1-Score": f1,
                "MSE": mse
            }
        
        # Mostrar resultados
        st.subheader("Resultados dos Modelos")
        
        # Criar dataframe de resultados
        results_df = pd.DataFrame({
            model: metrics
            for model, metrics in results.items()
        }).T
        
        st.write(results_df)
        
        # Gráfico de barras para F1-Score (melhor métrica para dados desbalanceados)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results_df.index, results_df["F1-Score"], color=["blue", "green", "orange"])
        ax.set_ylabel('F1-Score')
        ax.set_title('Comparação de Modelos - F1-Score (maior é melhor)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Visualizar coeficientes
        st.subheader("Coeficientes dos Modelos")
        
        coef_df = pd.DataFrame({
            'Feature': best_features,
            'Linear Regression': lr.coef_,
            f'Ridge (α={alpha_ridge})': ridge.coef_,
            f'Lasso (α={alpha_lasso})': lasso.coef_
        })
        
        st.write(coef_df.set_index('Feature'))
        
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
        ax.set_xlabel('Features')
        ax.set_ylabel('Coeficientes')
        ax.set_title('Importância das Features para Detecção de Fraudes')
        ax.set_xticks(index)
        ax.set_xticklabels(best_features, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Matriz de confusão para o melhor modelo
        st.subheader("Matriz de Confusão")
        
        # Encontrar o melhor modelo com base no F1-Score
        best_model_name = results_df["F1-Score"].idxmax()
        best_model_pred = predictions[best_model_name]
        
        cm = confusion_matrix(y_test, best_model_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
        ax.set_xlabel('Previsto')
        ax.set_ylabel('Real')
        ax.set_title(f'Matriz de Confusão - {best_model_name}')
        st.pyplot(fig)
        
        # Explicação sobre Ridge e Lasso para classificação
        st.subheader("Interpretação")
        
        st.write("""
        ### Aplicação de Ridge e Lasso para Detecção de Fraudes:
        
        1. **Interpretação dos Coeficientes**:
           - Coeficientes positivos: Indicam que valores maiores dessa feature aumentam a probabilidade de fraude
           - Coeficientes negativos: Indicam que valores maiores dessa feature diminuem a probabilidade de fraude
           - Coeficientes próximos a zero (especialmente em Lasso): Indicam features menos relevantes para a detecção
    
        2. **Comparação dos Modelos**:
           - **Regressão Linear**: Sem regularização, pode ser mais suscetível a overfitting, especialmente com muitas variáveis
           - **Ridge**: Reduz todos os coeficientes de forma proporcional, mantendo todas as features
           - **Lasso**: Tende a realizar seleção de features, eliminando algumas completamente (coeficientes = 0)
    
        3. **Por que usar regularização para fraudes?**
           - Dados de fraude geralmente têm muitas variáveis potencialmente correlacionadas
           - A regularização ajuda a evitar overfitting em dados de treinamento
           - Lasso pode identificar automaticamente as variáveis mais importantes para detecção
        """)
    
        # Adicionar thresholding interativo
        st.subheader("Ajuste de Limiar (Threshold)")
        
        st.write("""
        Em problemas de classificação desbalanceados como detecção de fraudes, 
        ajustar o limiar de decisão é crucial para equilibrar falsos positivos e falsos negativos.
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
        col1.metric("Acurácia", f"{custom_accuracy:.4f}")
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

    # Adicionar métodos de boosting
    st.subheader("🚀 Métodos de Boosting")

    st.write("""
    ### Métodos Ensemble: Boosting

    Algoritmos de Boosting constroem múltiplos modelos sequencialmente, onde cada modelo corrige 
    os erros de seus predecessores. Estes são particularmente eficazes para detecção de fraudes.

    - **AdaBoost**: Adaptive Boosting foca em instâncias classificadas incorretamente, aumentando seus pesos
    - **XGBoost**: eXtreme Gradient Boosting usa descida do gradiente para minimizar erros
    """)

    # Criar colunas para os dois modelos
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### AdaBoost")
        
        run_ada = st.checkbox("Treinar Modelo AdaBoost", value=False)
        if run_ada:
            from sklearn.ensemble import AdaBoostClassifier
            
            # Definir parâmetros
            n_estimators = st.slider("Número de Estimadores (AdaBoost)", 50, 300, 100)
            learning_rate = st.slider("Taxa de Aprendizado (AdaBoost)", 0.01, 2.0, 1.0, 0.01)
            
            with st.spinner("Treinando o modelo AdaBoost..."):
                # Iniciar contagem de tempo
                start = time.time()
                
                # Criar e treinar modelo
                ada_model = AdaBoostClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42
                )
                ada_model.fit(X_train_resampled, y_train_resampled)
                
                # Encerrar contagem de tempo
                duration = time.time() - start
                st.write(f"⏱️ Tempo de treinamento: {duration:.2f} segundos")
                
                # Fazer previsões
                y_pred_ada = ada_model.predict(X_test)
                
                # Calcular métricas
                ada_accuracy = accuracy_score(y_test, y_pred_ada)
                ada_precision = precision_score(y_test, y_pred_ada, zero_division=0)
                ada_recall = recall_score(y_test, y_pred_ada, zero_division=0)
                ada_f1 = f1_score(y_test, y_pred_ada, zero_division=0)
                
                # Exibir métricas
                st.metric("Acurácia", f"{ada_accuracy:.4f}")
                st.metric("Precisão", f"{ada_precision:.4f}")
                st.metric("Recall", f"{ada_recall:.4f}")
                st.metric("F1 Score", f"{ada_f1:.4f}")
                
                # Exibir matriz de confusão
                cm_ada = confusion_matrix(y_test, y_pred_ada)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
                ax.set_xlabel('Previsto')
                ax.set_ylabel('Real')
                ax.set_title('Matriz de Confusão AdaBoost')
                st.pyplot(fig)

    with col2:
        st.write("#### XGBoost")
        
        run_xgb = st.checkbox("Treinar Modelo XGBoost", value=False)
        if run_xgb:
            import xgboost as xgb
            
            # Definir parâmetros
            n_estimators_xgb = st.slider("Número de Estimadores (XGBoost)", 50, 300, 100)
            max_depth = st.slider("Profundidade Máxima (XGBoost)", 3, 10, 6)
            learning_rate_xgb = st.slider("Taxa de Aprendizado (XGBoost)", 0.01, 0.3, 0.1, 0.01)
            
            with st.spinner("Treinando o modelo XGBoost..."):
                # Iniciar contagem de tempo
                start = time.time()
                
                # Criar e treinar modelo
                xgb_model = xgb.XGBClassifier(
                    n_estimators=n_estimators_xgb,
                    max_depth=max_depth,
                    learning_rate=learning_rate_xgb,
                    random_state=42
                )
                xgb_model.fit(X_train_resampled, y_train_resampled)
                
                # Encerrar contagem de tempo
                duration = time.time() - start
                st.write(f"⏱️ Tempo de treinamento: {duration:.2f} segundos")
                
                # Fazer previsões
                y_pred_xgb = xgb_model.predict(X_test)
                
                # Calcular métricas
                xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
                xgb_precision = precision_score(y_test, y_pred_xgb, zero_division=0)
                xgb_recall = recall_score(y_test, y_pred_xgb, zero_division=0)
                xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
                
                # Exibir métricas
                st.metric("Acurácia", f"{xgb_accuracy:.4f}")
                st.metric("Precisão", f"{xgb_precision:.4f}")
                st.metric("Recall", f"{xgb_recall:.4f}")
                st.metric("F1 Score", f"{xgb_f1:.4f}")
                
                # Exibir matriz de confusão
                cm_xgb = confusion_matrix(y_test, y_pred_xgb)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'])
                ax.set_xlabel('Previsto')
                ax.set_ylabel('Real')
                ax.set_title('Matriz de Confusão XGBoost')
                st.pyplot(fig)
                
                # Importância das features para XGBoost
                fig, ax = plt.subplots(figsize=(8, 6))
                xgb.plot_importance(xgb_model, ax=ax, max_num_features=10)
                plt.title("Importância das Features - XGBoost")
                st.pyplot(fig)