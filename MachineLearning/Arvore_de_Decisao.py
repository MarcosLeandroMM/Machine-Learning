"""
As árvores de decisão são modelos de aprendizado de máquina que tomam decisões com base em regras hierárquicas. Cada nó na árvore representa uma decisão com base em uma característica, e os ramos (ramificações) conectam os nós. 
O objetivo é dividir o conjunto de dados em subconjuntos homogêneos (mais puros em termos de classe) da melhor maneira possível.

"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

# Carregando o conjunto de dados Iris
iris = load_iris()
X, Y = iris.data, iris.target

# Criando o modelo de árvore de decisão
modelo_arvore = DecisionTreeClassifier()
modelo_arvore.fit(X, Y)

# Extraindo regras da árvore de decisão
regras = export_text(modelo_arvore, feature_names=iris.feature_names)
print("Regras da Árvore de Decisão:\n", regras)

"""
Neste exemplo, uma árvore de decisão é treinada no conjunto de dados Iris para classificar as espécies de flores com base em suas características. As regras da árvore de decisão são extraídas e podem ser interpretadas para entender como as decisões são tomadas.



A linha regras = export_text(modelo_arvore, feature_names=iris.feature_names) extrai as regras da árvore de decisão treinada (modelo_arvore) em um formato de texto legível. Vamos quebrar os componentes dessa linha:

export_text: É uma função fornecida pelo scikit-learn que converte a estrutura de uma árvore de decisão em texto.

modelo_arvore: É o modelo de árvore de decisão treinado usando o conjunto de dados Iris.

feature_names=iris.feature_names: É um parâmetro opcional que fornece os nomes das características (ou atributos) usados no treinamento da árvore de decisão. No caso do conjunto de dados Iris, as características representam as medidas das pétalas e sépalas das flores.

iris.feature_names: É uma lista que contém os nomes das características do conjunto de dados Iris. No contexto deste conjunto de dados, iris.feature_names seria algo como ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], representando as dimensões das pétalas e sépalas.

Então, a linha de código completa está utilizando a função export_text para gerar um texto que descreve as regras da árvore de decisão, e o parâmetro feature_names é usado para associar nomes significativos às características ao invés de usar simplesmente índices.

"""