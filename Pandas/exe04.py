"""
Manipulação de Colunas:

Adicione uma nova coluna que seja uma combinação de duas colunas existentes.
Remova uma coluna desnecessária do DataFrame.
"""

import pandas as pd

meu_arquivo = 'dados.csv'

df = pd.read_csv(meu_arquivo)

# Adição de uma nova coluna:
df['Soma'] = df['Coluna1'] + df['Coluna2']


# Remoção de uma coluna desnecessária:

df = df.drop('ColunaDesnecessaria', axis=1)