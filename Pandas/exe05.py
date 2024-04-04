"""
Agrupamento e Agregação:

Agrupe o DataFrame por uma coluna específica.
Aplique uma função de agregação, como média ou soma, ao grupo.
"""
import pandas as pd 

meu_arquivo = 'dados.csv'

df = pd.read_csv(meu_arquivo)

# Agrupamento por uma coluna específica:
grupos = df.groupby('Categoria')

# Aplicação de uma função de agregação ao grupo:
# Exemplo com média:
media_por_grupos = grupos('Valor').mean()

# Exemplo com soma:
soma_por_grupos = grupos['Valor'].sum()


# aplicando várias funções de agregação usando o método agg
resultados_agregados = grupos.agg({
    'Valor': ['mean', 'sum'],
    'Quantidade': 'sum',
})