import pandas as pd
import numpy as np

# Carregando o dataset original
try:
    df = pd.read_csv('diagnosis2.csv', encoding='utf-16')
    print(f"Primeiras linhas do DataFrame:\n{df.head()}")
    print(f"Colunas do DataFrame: {df.columns}")
except FileNotFoundError:
    print("Arquivo não encontrado. Verifique o caminho e o nome do arquivo.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo: {e}")
    exit()

# Número de indivíduos a serem gerados
num_individuos = 1000

# Gerando novos indivíduos a partir dos dados existentes com amostragem com substituição
df_sintetico = df.sample(n=num_individuos, replace=True, random_state=42)

# Salvando o novo dataset em um arquivo CSV
df_sintetico.to_csv('sintetico_diagnosis.csv', index=False, encoding='utf-16')

print(f"Novo arquivo CSV com {num_individuos} indivíduos gerados salvo como 'sintetico_diagnosis.csv'")
