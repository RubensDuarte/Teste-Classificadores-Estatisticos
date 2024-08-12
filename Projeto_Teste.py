# Importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o dataset
try:
    df = pd.read_csv('novo_dataset_balanceado_Smote.csv', encoding='utf-8')
    print(f"Primeiras linhas do DataFrame:\n{df.head()}")
    print(f"Colunas do DataFrame: {df.columns}")
except FileNotFoundError:
    print("Arquivo não encontrado. Verifique o caminho e o nome do arquivo.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo: {e}")
    exit()

# Instanciando o scaler
scaler = MinMaxScaler()

# Normalizar os dados
df_norm = scaler.fit_transform(df.drop("Nephritis_of_Renal_Pelvis", axis=1))

# Balanceamento de classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(df_norm, df["Nephritis_of_Renal_Pelvis"])

# Divisão do conjunto de treino e teste com 20% para treino e 80% para teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.8, random_state=42)

# Verificando as dimensões dos conjuntos de treino e teste
print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")

# Verificar distribuição das classes no conjunto de treino
unique, counts = np.unique(y_train, return_counts=True)
print(f"Distribuição das classes no conjunto de treino: {dict(zip(unique, counts))}")

# Verificar distribuição das classes no conjunto de teste
unique, counts = np.unique(y_test, return_counts=True)
print(f"Distribuição das classes no conjunto de teste: {dict(zip(unique, counts))}")

# Treinamento dos classificadores
classificadores = {
    "MLP": MLPClassifier(max_iter=590, random_state=42),
    "RBF": SVC(kernel="rbf", probability=True, random_state=42),
    "SVM": SVC(kernel="linear", probability=True, random_state=42),
    "RF": RandomForestClassifier(random_state=42),
}

# Avaliação dos classificadores
resultados = {}
acuracias = {nome: {'Média': 0, 'Desvio Padrão': 0} for nome in classificadores.keys()}
conf_matrices = {}
roc_curves = {}

# Usando KFold com 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for nome, clf in classificadores.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Métricas de avaliação
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    resultados[nome] = {"Acurácia": acc, "Precisão": prec, "Recall": rec, "F1-score": f1}

    # Calculando a acurácia e o desvio padrão dos acertos usando KFold
    cv_scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring="accuracy")
    acuracia_media = np.mean(cv_scores)
    acuracia_std = np.std(cv_scores)
    acuracias[nome] = {'Média': acuracia_media, 'Desvio Padrão': acuracia_std}

    # Matriz de Confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrices[nome] = conf_matrix

    # Curva ROC
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_curves[nome] = (fpr, tpr)

# Criar DataFrame para os resultados de K-Fold
df_acuracias = pd.DataFrame(acuracias).T
print(df_acuracias.to_string())

# Criar DataFrame para os resultados de acurácia
df_resultados = pd.DataFrame(resultados).T
print(df_resultados.to_string())

# Visualização dos resultados
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Loop pelos algoritmos
for ax, (nome, clf) in zip(axes, classificadores.items()):
    # Matriz de Confusão
    sns.heatmap(conf_matrices[nome], annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(f"Matriz de Confusão - {nome}")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for nome, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=nome)

plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC")
plt.legend()
plt.show()

# Gráfico de barras com acurácia média e desvio padrão
plt.figure(figsize=(10, 6))
df_acuracias.plot(kind='bar', y=['Média', 'Desvio Padrão'], capsize=4, rot=45, ax=plt.gca())
plt.ylabel("Acurácia")
plt.title("Acurácia Média e Desvio Padrão dos Modelos")
plt.show()

# Gráfico de barras com a acurácia de cada modelo
plt.figure(figsize=(10, 6))
df_resultados['Acurácia'].plot(kind='bar', color='skyblue', capsize=4)
plt.ylabel("Acurácia")
plt.title("Acurácia de Cada Modelo")
plt.show()

# Criando um DataFrame para armazenar os resultados de cada indivíduo
resultados_individuais = pd.DataFrame({
    "Indivíduo": range(1, len(y_test) + 1),
    "MLP": classificadores["MLP"].predict(X_test),
    "RBF": classificadores["RBF"].predict(X_test),
    "SVM": classificadores["SVM"].predict(X_test),
    "RF": classificadores["RF"].predict(X_test),
    "Real": y_test
})

# Exibindo a planilha na tela
print("\nResultados individuais dos modelos:\n")
print(resultados_individuais.to_string(index=False))

# Salvando os resultados em um arquivo de texto
with open('resultados_classificadores.txt', 'w') as file:
    file.write("Acurácias:\n")
    file.write(df_acuracias.to_string())
    file.write("\n\nResultados:\n")
    file.write(df_resultados.to_string())

    file.write("\n\nMatrizes de Confusão:\n")
    for nome, conf_matrix in conf_matrices.items():
        file.write(f"\n{nome}:\n")
        file.write(np.array2string(conf_matrix))

    file.write("\n\nCurvas ROC:\n")
    for nome, (fpr, tpr) in roc_curves.items():
        file.write(f"\n{nome}:\n")
        file.write("FPR: " + np.array2string(fpr) + "\n")
        file.write("TPR: " + np.array2string(tpr) + "\n")

    # Salvando resultados individuais dos modelos
    file.write("\n\nResultados Individuais dos Modelos:\n")
    file.write(resultados_individuais.to_string(index=False))

print("Resultados salvos no arquivo 'resultados_classificadores.txt'")
