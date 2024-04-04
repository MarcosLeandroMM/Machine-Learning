"""
Agrupamento Avançado:

Realize uma operação de agrupamento que envolva várias funções de agregação, como média, soma e contagem.
"""

import pandas as pd 

# Criando um DataFrame de exemplo

dados = {'Categoria': ['A', 'B', 'A', 'B', 'A', 'B'],
         'Valor': [10, 20, 15, 25, 12, 18],
         'Quantidade': [50, 30, 40, 20, 45, 35]}

df = pd.DataFrame(dados)

resultados_agregados = df.groupby('Categoria').agg({
    'Valor': ['mean', 'sum'],
    'Quantidade': ['sum', 'count']
}).reset_index()

# Renomeando as colunas para maior clareza
resultados_agregados.columns = ['Categoria', 'Média_Valor', 'Soma_Valor', 'Soma_Quantidade', 'Contagem_Quantidade']

print(resultados_agregados)