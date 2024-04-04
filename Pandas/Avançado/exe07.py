"""
Trabalhando com Dados Hierárquicos:

Manipule um DataFrame com índices e colunas hierárquicos de forma avançada.

"""

import pandas as pd
import numpy as np

# Criando um DataFrame com índices e colunas hierárquicos
indices = pd.MultiIndex.from_product([['A', 'B'], ['X', 'Y']], names=['Grupo', 'Subgrupo'])
colunas = pd.MultiIndex.from_product([['Valor', 'Quantidade'], ['Total', 'Média']], names=['Tipo', 'Estatística'])

df = pd.DataFrame(np.random.randn(4, 4), index=indices, columns=colunas)

# Exibindo o DataFrame com índices e colunas hierárquicos
print("DataFrame com Índices e Colunas Hierárquicos:")
print(df)

# Acessando dados usando índices hierárquicos
valor_A_X = df.loc[('A', 'X'), ('Valor', 'Total')]
print("\nValor para ('A', 'X') na coluna ('Valor', 'Total'):", valor_A_X)

# Realizando operações com índices hierárquicos
soma_por_grupo = df.sum(level='Grupo', axis=0)
print("\nSoma por Grupo:")
print(soma_por_grupo)


"""
pd.MultiIndex.from_product é utilizado para criar índices e colunas hierárquicos.

df.loc[('A', 'X'), ('Valor', 'Total')] é usado para acessar um valor específico no DataFrame com índices e colunas hierárquicos.

df.sum(level='Grupo', axis=0) é utilizado para realizar a soma por nível hierárquico (no caso, o nível 'Grupo').
"""