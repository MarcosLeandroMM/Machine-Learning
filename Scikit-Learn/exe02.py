"""Exercício 2: Pré-processamento de Dados

Utilize o StandardScaler do Scikit-learn para padronizar os dados do conjunto de treinamento.
"""

from sklearn.preprocessing import StandardScaler

# Suponha que você tenha um conjunto de treinamento chamado X_train

# Criar uma instância do StandardScaler
scaler = StandardScaler()

# Ajustar o scaler aos dados de treinamento e transformar os dados
X_train_scaled = scaler.fit_transform(X_train)

'''
O StandardScaler do scikit-learn é uma ferramenta útil para padronizar os dados, o que significa que ele os transforma de modo que tenham uma média zero e um desvio padrão de 1. Isso é importante em muitos algoritmos de aprendizado de máquina, especialmente aqueles que são sensíveis à escala dos dados.

 O método fit_transform ajusta o scaler aos dados e, em seguida, os transforma. O resultado, X_train_scaled, é o conjunto de treinamento padronizado.

É importante mencionar que, ao usar o fit_transform no conjunto de treinamento, você deve usar o mesmo scaler para transformar conjuntos de teste ou dados de produção. Isso pode ser feito usando o método transform, como mostrado abaixo:

'''

# Suponha que você tenha um conjunto de teste chamado X_test

# Transformar os dados de teste usando o mesmo scaler
X_test_scaled = scaler.transform(X_test)
