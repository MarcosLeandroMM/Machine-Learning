"""
Exercício 1: Introdução ao Scikit-learn

Importe o conjunto de dados Iris usando load_iris do sklearn.datasets.
Divida o conjunto de dados em conjuntos de treinamento e teste.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dimensoes do conjunto de treinamento:", X_train.shape, y_train.shape)

print("Dimensoes do conjunto de teste:", X_test.shape, y_test.shape)

"""

Conjunto de Treinamento (X_train, y_train):

X_train: Contém as características (atributos) dos exemplos de treinamento. Cada linha representa um exemplo, e cada coluna representa um atributo. No caso, há 120 exemplos de treinamento e 4 atributos para cada exemplo (dimensões: 120 x 4).
y_train: Contém os rótulos (classes) correspondentes aos exemplos de treinamento. Há um rótulo para cada exemplo de treinamento. No caso, há 120 rótulos (dimensão: 120).
Conjunto de Teste (X_test, y_test):

X_test: Similar a X_train, contém as características dos exemplos de teste. No entanto, este conjunto é usado para avaliar o desempenho do modelo após o treinamento. Há 30 exemplos de teste e 4 atributos para cada exemplo (dimensões: 30 x 4).
y_test: Similar a y_train, contém os rótulos correspondentes aos exemplos de teste. Há um rótulo para cada exemplo de teste. No caso, há 30 rótulos (dimensão: 30).
Portanto, a saída indica que o conjunto de treinamento possui 120 exemplos, cada um com 4 atributos, e o conjunto de teste possui 30 exemplos, também com 4 atributos. Os rótulos (y) associados indicam as classes a que pertencem os exemplos, sendo 120 no conjunto de treinamento e 30 no conjunto de teste.


Treinamento:
Objetivo: Durante a fase de treinamento, o modelo é exposto a um conjunto de dados rotulado, chamado conjunto de treinamento. Esse conjunto inclui exemplos onde as entradas (características) estão associadas aos rótulos (saídas ou classes corretas). O modelo utiliza esses dados para aprender padrões e relações entre as características e os rótulos.

Processo de Treinamento: Durante o treinamento, o modelo ajusta seus parâmetros internos de acordo com os padrões presentes nos dados de treinamento. O objetivo é fazer com que o modelo generalize bem para dados não vistos, ou seja, que ele possa fazer previsões precisas para dados que não foram usados durante o treinamento.

Teste:
Objetivo: Após o treinamento, é necessário avaliar o desempenho do modelo em dados que ele nunca viu antes. Para isso, utiliza-se um conjunto separado chamado conjunto de teste. O conjunto de teste é crucial para verificar se o modelo é capaz de generalizar adequadamente, ou seja, se ele consegue fazer previsões precisas em situações não vistas durante o treinamento.

Processo de Teste: O modelo faz previsões no conjunto de teste e essas previsões são comparadas com os rótulos verdadeiros. As métricas de desempenho, como acurácia, precisão, recall, F1-score, entre outras, são calculadas para avaliar o quão bem o modelo está se saindo em dados não vistos. Essas métricas indicam a capacidade do modelo de generalizar para novos dados.

Por que Treinar e Testar?

Overfitting e Underfitting: O treinamento e teste ajudam a lidar com problemas de overfitting (quando o modelo se ajusta demais aos dados de treinamento e não generaliza bem) e underfitting (quando o modelo é muito simples para capturar os padrões nos dados).

Avaliação do Desempenho:

Conjunto de Treinamento: Usado para ensinar o modelo.
Conjunto de Teste: Usado para avaliar o desempenho e a capacidade de generalização do modelo.
Generalização: O objetivo final é ter um modelo que generalize bem para dados não vistos, fazendo previsões precisas em situações do mundo real.

Preparação para Previsões:

Agora, você pode usar o modelo treinado para fazer previsões em novos dados. Antes de fazer isso, certifique-se de que os novos dados tenham a mesma estrutura e formato que os dados de treinamento, para que o modelo possa processá-los corretamente.
Função de Previsão:

A maioria dos frameworks de aprendizado de máquina fornece uma função de previsão que você pode usar para obter as previsões do seu modelo. Por exemplo, no Scikit-learn, após treinar o modelo, você pode usar o método predict para fazer previsões.
Exemplo no Scikit-learn:


# Treinamento do modelo (supondo que clf seja seu modelo)
clf.fit(X_train, y_train)

# Fazendo previsões em novos dados (X_new)
predictions = clf.predict(X_new)
Avaliação das Previsões:

Após obter as previsões, você pode avaliar a qualidade delas comparando-as com os rótulos verdadeiros (se disponíveis). Isso pode ser feito usando as mesmas métricas de desempenho que foram usadas durante o teste.
Lembre-se de que a qualidade das previsões em novos dados depende da qualidade do treinamento do modelo e da capacidade do modelo de generalizar padrões para dados não vistos. Se o modelo foi treinado e testado adequadamente, ele deve ser capaz de fornecer previsões razoáveis em novos dados.







"""