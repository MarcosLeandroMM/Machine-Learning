"""
Utilize um conjunto de dados apropriado para o agrupamento hierárquico.
Experimente diferentes métodos de ligação e visualize dendrogramas.


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage

# Carregando o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Realizando o agrupamento hierárquico com diferentes métodos de ligação
methods = ['single', 'complete', 'average', 'ward']

plt.figure(figsize=(15, 10))

for i, method in enumerate(methods, 1):
    plt.subplot(2, 2, i)
    
    # Calculando a matriz de ligação
    Z = linkage(X, method)
    
    # Criando e exibindo o dendrograma
    dendrogram(Z, labels=y, leaf_rotation=90, leaf_font_size=8)
    plt.title(f'Dendrograma - Método de Ligação: {method.capitalize()}')

plt.tight_layout()
plt.show()


"""
Carregamos o conjunto de dados Iris.
Utilizamos diferentes métodos de ligação (single, complete, average, ward) para realizar o agrupamento hierárquico.
Criamos e exibimos dendrogramas para cada método de ligação.
Experimente diferentes métodos de ligação e observe como eles afetam a estrutura do dendrograma. O método ward é comumente utilizado, especialmente quando os clusters têm variâncias diferentes. O método average é sensível a diferentes formas e tamanhos de clusters. O método complete pode ser sensível a outliers. O método single tende a formar clusters alongados.


"""