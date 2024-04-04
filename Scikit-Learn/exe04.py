"""

Exercício 4: Classificação com SVM

Utilize o SVM (Support Vector Machine) para classificar o conjunto de dados Iris.
Experimente diferentes kernels (linear, polinomial, radial) e ajuste os parâmetros.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Carregar o conjunto de dados Iris
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados (importante para SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar um classificador SVM
svm_classifier = SVC()

# Definir os parâmetros que você deseja ajustar
parameters = {'kernel': ['linear', 'poly', 'rbf'],
              'C': [0.1, 1, 10],
              'gamma': [0.1, 1, 'auto']}

# Usar GridSearchCV para encontrar os melhores parâmetros
grid_search = GridSearchCV(svm_classifier, parameters, cv=5)
grid_search.fit(X_train, y_train)

# Imprimir os melhores parâmetros encontrados
print("Melhores parâmetros:", grid_search.best_params_)

# Fazer previsões no conjunto de teste
y_pred = grid_search.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')
