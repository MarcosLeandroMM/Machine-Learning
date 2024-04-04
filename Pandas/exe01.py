"""
Leitura e Visualização de Dados:

Leia um arquivo CSV ou Excel usando o Pandas.
Exiba as primeiras linhas do DataFrame.
"""

import pandas as pd

meu_arquivo = 'dados.csv'

try:
    df = pd.read_csv(meu_arquivo)

except FileNotFoundError:
    print(f"O arquivo {meu_arquivo} não foi encontrado.")

if 'df' in locals():
    print("As primeiras linhas do DataFrame:")
    print(df.head())

