"""
Agregação Customizada:

Crie uma função de agregação personalizada e aplique-a a um grupo no DataFrame.


A agregação personalizada no Pandas permite que você crie funções personalizadas e as aplique a grupos específicos em um DataFrame usando o método agg. Vamos considerar um exemplo em que temos um DataFrame com informações de vendas e queremos calcular a soma ponderada do preço de venda pelo número de unidades vendidas para cada grupo de produtos:
"""

import pandas as pd

# Criando um DataFrame de exemplo
dados = {
    'Produto': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Preco': [10, 12, 15, 18, 11, 16],
    'Quantidade': [100, 150, 80, 120, 200, 90]
}

df = pd.DataFrame(dados)

# Definindo uma função de agregação personalizada para calcular a soma ponderada
def soma_ponderada(x):
    return (x['Preco'] * x['Quantidade']).sum() / x['Quantidade'].sum()

# Aplicando a função de agregação personalizada ao grupo de produtos
resultado_agregacao = df.groupby('Produto').agg(Soma_Ponderada=('Produto', soma_ponderada))

# Exibindo o resultado da agregação
print(resultado_agregacao)


"""

A agregação personalizada no Pandas permite que você crie funções personalizadas e as aplique a grupos específicos em um DataFrame usando o método agg. Vamos considerar um exemplo em que temos um DataFrame com informações de vendas e queremos calcular a soma ponderada do preço de venda pelo número de unidades vendidas para cada grupo de produtos:

python
Copy code
import pandas as pd

# Criando um DataFrame de exemplo
dados = {
    'Produto': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Preco': [10, 12, 15, 18, 11, 16],
    'Quantidade': [100, 150, 80, 120, 200, 90]
}

df = pd.DataFrame(dados)

# Definindo uma função de agregação personalizada para calcular a soma ponderada
def soma_ponderada(x):
    return (x['Preco'] * x['Quantidade']).sum() / x['Quantidade'].sum()

# Aplicando a função de agregação personalizada ao grupo de produtos
resultado_agregacao = df.groupby('Produto').agg(Soma_Ponderada=('Produto', soma_ponderada))

# Exibindo o resultado da agregação
print(resultado_agregacao)
Neste exemplo, a função de agregação personalizada soma_ponderada calcula a soma ponderada do preço pelo número de unidades para cada grupo de produtos. A função é então aplicada ao grupo usando o método agg.

"""