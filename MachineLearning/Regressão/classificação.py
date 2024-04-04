# Classificação com Regressão Logística
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Dados de exemplo
X, Y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_classes=2, random_state=42)

# Divisão dos dados
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

# Criando e treinando o modelo de regressão logística
modelo_logistico = LogisticRegression()
modelo_logistico.fit(X_treino, Y_treino)

# Prevendo classes
Y_predito = modelo_logistico.predict(X_teste)

# Avaliação do modelo
acuracia = accuracy_score(Y_teste, Y_predito)
matriz_confusao = confusion_matrix(Y_teste, Y_predito)
print(f'Acurácia: {acuracia}')
print(f'Matriz de Confusão:\n{matriz_confusao}')

# Visualização da Matriz de Confusão
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()
