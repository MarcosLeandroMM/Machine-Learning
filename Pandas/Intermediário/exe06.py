"""
Trabalhando com Dados Categóricos:

Converta uma coluna em dados categóricos e explore as funcionalidades relacionadas.

Converter uma coluna em dados categóricos no Pandas pode ser útil para economizar memória e fornecer recursos específicos para trabalhar com categorias. 
"""

import pandas as pd

# Criando um DataFrame de exemplo
dados = {'Categoria': ['A', 'B', 'A', 'C', 'B', 'C']}
df = pd.DataFrame(dados)

# Convertendo a coluna 'Categoria' para dados categóricos
df['Categoria'] = df['Categoria'].astype('category')

# Exibindo as categorias únicas
categorias_unicas = df['Categoria'].cat.categories
print("Categorias Únicas:", categorias_unicas)

# Contando a frequência de cada categoria
contagem_categorias = df['Categoria'].value_counts()
print("\nContagem de Categorias:")
print(contagem_categorias)

# Renomeando categorias
df['Categoria'] = df['Categoria'].cat.rename_categories({'A': 'Categoria_A', 'B': 'Categoria_B', 'C': 'Categoria_C'})

# Exibindo o DataFrame resultante
print("\nDataFrame com Categorias Renomeadas:")
print(df)


"""
A coluna 'Categoria' é convertida para dados categóricos usando o método astype('category').
O atributo cat.categories é usado para obter as categorias únicas.
O método value_counts() é utilizado para contar a frequência de cada categoria.
As categorias podem ser renomeadas usando o método cat.rename_categories().

"""