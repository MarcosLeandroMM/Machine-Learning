"""
Operações Estatísticas:

Calcule estatísticas descritivas, como média, mediana e desvio padrão, para colunas numéricas.

"""

import pandas as pd 

meu_arquivo = 'dados.csv'

df = pd.read_csv(meu_arquivo)

nome_coluna_numerica = 'ColunaNumerica'

# Média
media = df[nome_coluna_numerica].mean()

# Mediana
mediana = df[nome_coluna_numerica].median()

# Desvio Padrão
desvio_padrao = df[nome_coluna_numerica].std()