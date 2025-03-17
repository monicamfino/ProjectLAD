import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# Carregar o dataset (substitua pelo caminho correto do arquivo)
df = pd.read_csv("creditcard.csv")

def save_and_show_plot(filename):
    plt.savefig(f'plots/{filename}', bbox_inches='tight')
    plt.show()


df = df.dropna()
df = df.fillna(df.mean())


# Exibir as primeiras linhas
print(df.head())

# Calcular média e mediana das principais variáveis
print("Média:")
print(df.mean())

print("\nMediana:")
print(df.median())

# Variância e desvio padrão
print("\nVariância:")
print(df.var())

print("\nDesvio Padrão:")
print(df.std())

# Matriz de covariância
print("\nMatriz de Covariância:")
print(df.cov())

# Matriz de correlação
print("\nMatriz de Correlação:")
print(df.corr())

# Visualizar a matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Matriz de Correlação")
save_and_show_plot("matriz_de_correlação.png")


# Separando os dados
fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0]

# Comparando a média do valor da transação
print("Média do valor da transação - Fraude:", fraud["Amount"].mean())
print("Média do valor da transação - Legítima:", legit["Amount"].mean())

plt.figure(figsize=(12, 6))

# Gráficos com transparência e normalização
sns.histplot(fraud["Amount"], bins=50, color="red", label="Fraude", kde=True, alpha=0.6, stat="density")
sns.histplot(legit["Amount"], bins=50, color="blue", label="Legítima", kde=True, alpha=0.6, stat="density")

# Escala logarítmica no eixo x
plt.xscale("log")

# Linhas de mediana para melhor contexto
plt.axvline(fraud["Amount"].median(), color="red", linestyle="dashed", label="Mediana Fraude")
plt.axvline(legit["Amount"].median(), color="blue", linestyle="dashed", label="Mediana Legítima")

# Rótulos e título aprimorados
plt.xlabel("Valor da Transação (Escala Log)")
plt.ylabel("Densidade")
plt.title("Distribuição dos Valores de Transação (Fraude vs Legítima)")

# Ajustando os limites dos eixos
plt.xlim(1, 10000)  # Ajuste do eixo X
plt.ylim(0, 0.03)   # Ajuste do eixo Y

plt.legend()

# Salvar e exibir gráfico
save_and_show_plot("distribuicao_valores_transacao_ajustada.png")
