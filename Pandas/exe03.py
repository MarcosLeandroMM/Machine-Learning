"""
Filtragem de Dados:

Filtrar o DataFrame para incluir apenas linhas que satisfaçam uma condição específica.

"""

import pandas as pd

meu_arquivo = 'dados.csv'

df = pd.read_csv(meu_arquivo)

condicao = df['Idade'] > 25
# usando o método loc
df_filtrado = df.loc[condicao]
# ou diretamente no dataframe
df_filtrado = df[condicao]
