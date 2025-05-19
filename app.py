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
        
        # Treinar modelo
        with st.spinner('Treinando o modelo...'):
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
        
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
        st.subheader("Relatório de Classificação")
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
    
    with model_tabs[2]:
        st.markdown("## Ridge e Lasso Regression para Detecção de Fraudes")
        
        st.write("""
        ### Regressão Regularizada para Classificação de Fraudes
        
        Embora Ridge e Lasso sejam técnicas de regressão, elas podem ser aplicadas para problemas de classificação 
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
        
        # Métricas com threshold personalizado
        custom_accuracy = accuracy_score(y_test, y_pred_custom)
        custom_precision = precision_score(y_test, y_pred_custom, zero_division=0)
        custom_recall = recall_score(y_test, y_pred_custom, zero_division=0)
        custom_f1 = f1_score(y_test, y_pred_custom, zero_division=0)
        
        # Mostrar métricas
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
        ax.set_title(f'Matriz de Confusão com Limiar = {custom_threshold}')
        st.pyplot(fig)
