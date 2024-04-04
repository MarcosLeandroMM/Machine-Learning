""" O que é ensemble
 Ensemble refere-se a uma técnica que combina múltiplos modelos de aprendizado de máquina para melhorar o desempenho geral do sistema. Em vez de confiar em um único modelo para fazer previsões ou classificações, o Ensemble utiliza a sabedoria das multidões, combinando as previsões de vários modelos individuais para chegar a uma previsão final mais precisa e robusta.

 Votação (Voting): Os modelos individuais fazem previsões e a previsão final é determinada por votação majoritária. Por exemplo, no caso da classificação, a classe mais frequentemente prevista pelos modelos individuais é selecionada como a classe final.

Média (Averaging): Os modelos individuais fazem previsões numéricas e a previsão final é a média dessas previsões. Isso é comum em problemas de regressão.

Bagging (Bootstrap Aggregating): Os modelos individuais são treinados em conjuntos de dados de treinamento diferentes, criados por amostragem com substituição do conjunto de dados original. A previsão final é a média ou a votação dos modelos treinados.

Boosting: Os modelos individuais são treinados sequencialmente, com cada novo modelo focando nas instâncias que foram classificadas incorretamente pelos modelos anteriores. A previsão final é ponderada com base na precisão dos modelos individuais.

Stacking: Os modelos individuais fazem previsões que são usadas como entradas para um meta-modelo, que então faz a previsão final. Isso pode ser feito em várias camadas.


"""


"""
Exercício 10: Ensemble Learning

Experimente o método de ensemble, como o VotingClassifier ou BaggingClassifier, em um conjunto de dados de classificação.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar classificadores individuais
decision_tree = DecisionTreeClassifier(random_state=42)
knn_classifier = KNeighborsClassifier()
logistic_regression = LogisticRegression(max_iter=1000, random_state=42)

# Criar o ensemble usando o VotingClassifier
ensemble_classifier = VotingClassifier(estimators=[
    ('decision_tree', decision_tree),
    ('knn', knn_classifier),
    ('logistic_regression', logistic_regression)
], voting='hard')  # 'hard' para voto por maioria, 'soft' para votação ponderada pelas probabilidades

# Treinar o ensemble
ensemble_classifier.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = ensemble_classifier.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do ensemble: {accuracy}')

'''

Neste exemplo, estamos usando um ensemble que combina um classificador de árvore de decisão (DecisionTreeClassifier), um classificador KNN (KNeighborsClassifier), e uma regressão logística (LogisticRegression). O VotingClassifier usa a estratégia de voto por maioria (voting='hard'), onde a classe predita é a classe que recebe a maioria dos votos dos classificadores individuais.

'''