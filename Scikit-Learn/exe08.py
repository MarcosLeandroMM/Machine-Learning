"""
SVM (Support Vector Machine) é um algoritmo de aprendizado supervisionado usado para classificação e regressão. O objetivo do SVM é encontrar o hiperplano de separação que melhor divide os dados em classes distintas. O hiperplano é escolhido de forma que a distância entre ele e os pontos mais próximos de cada classe, chamados de vetores de suporte, seja maximizada.

O GridSearch é uma técnica usada para encontrar os melhores hiperparâmetros para um modelo de aprendizado de máquina. Hiperparâmetros são parâmetros que não são aprendidos diretamente do processo de treinamento do modelo e devem ser definidos pelo usuário antes do treinamento. O GridSearch realiza uma busca exaustiva em uma grade de valores especificados para cada hiperparâmetro do modelo, avaliando o desempenho do modelo para cada combinação de valores de hiperparâmetros e selecionando aqueles que produzem o melhor desempenho.

Um exemplo de aplicação de SVM com GridSearch no mundo real é na classificação de texto para análise de sentimentos em mídias sociais. Suponha que você tenha um grande conjunto de dados de tweets rotulados como positivos, negativos ou neutros e queira treinar um classificador SVM para identificar o sentimento de novos tweets. Você pode usar o GridSearch para encontrar os melhores valores de hiperparâmetros para o SVM, como o tipo de kernel (linear, polinomial, RBF), a regularização (parâmetro C) e o coeficiente do kernel (para kernels polinomiais e RBF). Depois de encontrar os melhores hiperparâmetros, você pode treinar o modelo SVM com esses valores e usá-lo para classificar novos tweets com base em seus sentimentos.

"""


"""
Exercício 8: Ajuste de Hiperparâmetros com Grid Search

Utilize o GridSearchCV para encontrar os melhores hiperparâmetros para um modelo específico.

"""

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Iris
iris = load_iris()
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

'''
Neste exemplo, estamos usando o conjunto de dados Iris, dividindo-o em conjuntos de treinamento e teste, padronizando os dados, criando um classificador SVM e utilizando o GridSearchCV para encontrar os melhores hiperparâmetros.

Lembre-se de que você pode ajustar os parâmetros e o modelo conforme necessário para o seu problema específico. O GridSearchCV realiza uma busca exaustiva pelos melhores parâmetros dentro do espaço definido, tornando-o uma ferramenta útil para otimização de hiperparâmetros. 

'''