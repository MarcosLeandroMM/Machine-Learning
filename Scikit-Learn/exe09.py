"""
A Análise de Componentes Principais (PCA) é uma técnica de redução de dimensionalidade amplamente utilizada em aprendizado de máquina e estatística. O objetivo do PCA é reduzir a dimensionalidade de um conjunto de dados, preservando ao máximo a variação dos dados. Isso é feito transformando os dados originais em um novo conjunto de variáveis chamadas de componentes principais, que são combinações lineares das variáveis originais.

O PCA é útil em situações em que você tem um grande número de variáveis ​​(ou dimensões) em seus dados e deseja reduzi-las para um número menor de variáveis, enquanto ainda retém o máximo possível de informação. Isso pode ser benéfico para várias razões, incluindo a simplificação da visualização dos dados, a remoção de multicolinearidade entre as variáveis, a redução do tempo de treinamento de modelos de aprendizado de máquina e a melhoria da precisão do modelo.

Um exemplo de aplicação do PCA no mundo real é na análise de dados biomédicos. Considere um conjunto de dados que contém várias medidas biomédicas (por exemplo, pressão arterial, taxa de glicose, níveis de colesterol, etc.) para um grande número de pacientes. Se quisermos entender as principais fontes de variação nos dados e reduzir a dimensionalidade para facilitar a visualização e a interpretação, poderíamos aplicar o PCA. Isso nos permitiria identificar os padrões subjacentes nos dados e visualizá-los em um espaço de dimensões reduzidas, mantendo a maior parte da variação original.


"""

"""
Exercício 9: Redução de Dimensionalidade com PCA

Aplique a Análise de Componentes Principais (PCA) a um conjunto de dados.
Avalie a quantidade de informação retida ao reduzir as dimensões.

"""

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Carregar o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Aplicar PCA para reduzir a dimensionalidade
pca = PCA()
X_pca = pca.fit_transform(X)

# Plotar a variância explicada acumulada
explained_variance_ratio_cumulative = pca.explained_variance_ratio_.cumsum()

plt.plot(range(1, len(explained_variance_ratio_cumulative) + 1), explained_variance_ratio_cumulative, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada Acumulada')
plt.title('Análise de Componentes Principais (PCA)')
plt.show()

'''
Neste código, estamos aplicando o PCA ao conjunto de dados Iris e plotando a variância explicada acumulada em relação ao número de componentes principais. Isso nos dará uma ideia de quanto da variabilidade original dos dados é retida ao reduzir as dimensões.

Você pode observar o gráfico e decidir quantos componentes principais manter com base na quantidade de informação que você deseja reter. Em muitos casos, um valor em torno de 95% ou 99% de variância explicada acumulada pode ser escolhido, mas a escolha específica depende do seu contexto e requisitos.

Lembre-se de que a redução de dimensionalidade é uma ferramenta poderosa, mas deve ser usada com cuidado, pois pode resultar na perda de informações importantes.

'''