"""
Tratamento de Dados Ausentes:

Identifique e lide com valores ausentes no DataFrame.

Para identificar e lidar com valores ausentes (NaN) no DataFrame, você pode usar métodos do Pandas, como isna(), fillna(), e dropna(). 

"""

import pandas as pd

meu_arquivo = 'dados.csv'

df = pd.read_csv(meu_arquivo)

# Identificação de valores ausentes
print(df.isna().any())

# Conta o número de valores ausentes em cada coluna
print(df.isna().sum())


# Tratamento de valores ausentes:
# Remoção de linhas ou colunas com valores ausentes:
df_sem_nan_linhas = df.dropna()

# Remoção de colunas
df_sem_nan_colunas = df.dropna(axis=1)


# Preenche todos os valores ausentes com a média da coluna
media_colunas = df.mean()
df_preenchido_media = df.fillna(media_colunas)


# Preenche todos os valores ausentes na coluna 'Idade' com a média da coluna
media_idade = df['Idade'].mean()
df['Idade'] = df['Idade'].fillna(media_idade)
