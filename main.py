import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o dataset (substitua pelo caminho correto do arquivo)
df = pd.read_csv("creditcard.csv")

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
plt.show()

# Separando os dados
fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0]

# Comparando a média do valor da transação
print("Média do valor da transação - Fraude:", fraud["Amount"].mean())
print("Média do valor da transação - Legítima:", legit["Amount"].mean())

# Comparando a distribuição dos valores de transação
plt.figure(figsize=(10,5))
sns.histplot(fraud["Amount"], bins=50, color="red", label="Fraude", kde=True)
sns.histplot(legit["Amount"], bins=50, color="blue", label="Legítima", kde=True)
plt.legend()
plt.title("Distribuição dos valores de transação")
plt.show()

df["Rolling_Mean_Amount"] = df["Amount"].rolling(window=5).mean()

df["Std_Amount"] = df["Amount"].rolling(window=5).std()

df["Hour"] = (df["Time"] // 3600) % 24  # Convertendo o tempo para horas do dia

df["Amount_Category"] = pd.cut(df["Amount"], bins=[0, 10, 50, 100, 500, 5000, np.inf],
                               labels=["Very Low", "Low", "Medium", "High", "Very High", "Extreme"])

df["Delta_Amount"] = df["Amount"].diff()

# Comparação da distribuição do valor da transação por fraude e não fraude
plt.figure(figsize=(10,5))
sns.boxplot(x=df["Class"], y=df["Amount"])
plt.title("Distribuição do Valor da Transação por Classe (0 = Legítima, 1 = Fraude)")
plt.show()

# Visualizar a média do valor das transações ao longo do tempo
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x="Hour", y="Amount", hue="Class", estimator="mean")
plt.title("Média do Valor da Transação ao Longo do Dia")
plt.show()


