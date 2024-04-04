

"""
Exercício: Otimização Bayesiana de Hiperparâmetros:
Em vez de Grid Search, aplique otimização bayesiana para ajustar hiperparâmetros de um modelo.

A otimização bayesiana de hiperparâmetros é uma abordagem eficaz para encontrar os melhores conjuntos de hiperparâmetros para um modelo sem a necessidade de realizar uma busca exaustiva, como é feito no Grid Search. A biblioteca scikit-optimize (também conhecida como skopt) é uma opção popular para implementar otimização bayesiana em Python

A otimização bayesiana é uma abordagem mais eficiente para ajustar hiperparâmetros de modelos de aprendizado de máquina em comparação com a busca em grade. Ela utiliza técnicas probabilísticas para encontrar os melhores conjuntos de hiperparâmetros, explorando de forma inteligente o espaço de busca com base nas observações anteriores do desempenho do modelo.

"""

from skopt import BayesSearchCV
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Carregar conjunto de dados de exemplo (Boston Housing Dataset)
X, y = load_boston(return_X_y=True)

# Dividir conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir o modelo
model = SVR()

# Definir a grade de hiperparâmetros para otimização bayesiana
param_grid = {
    'C': (1e-6, 1e+6, 'log-uniform'),  # Parâmetro de regularização C
    'gamma': (1e-6, 1e+1, 'log-uniform'),  # Parâmetro de kernel gamma
    'epsilon': (1e-6, 1e+1, 'log-uniform')  # Parâmetro de folga epsilon
}

# Inicializar otimizador bayesiano
opt = BayesSearchCV(model, param_grid, cv=5, n_iter=50, random_state=42)

# Realizar a busca de hiperparâmetros
opt.fit(X_train, y_train)

# Exibir os melhores hiperparâmetros encontrados
print("Best parameters found: ", opt.best_params_)

# Avaliar o modelo no conjunto de teste
print("Test set score: {:.2f}".format(opt.score(X_test, y_test)))
