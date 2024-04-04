"""
Melt e Pivot:

Utilize as funções melt e pivot para remodelar o DataFrame.

As funções melt e pivot são úteis para remodelar DataFrames no Pandas. 
"""

"""
A função melt é utilizada para converter um DataFrame mais largo em um mais longo, mantendo algumas colunas fixas e "derretendo" outras colunas para criar uma nova coluna de variáveis e uma nova coluna de valores.

"""

import pandas as pd

# Criando um DataFrame de exemplo
df_largo = pd.DataFrame({
    'Nome': ['Alice', 'Bob', 'Charlie'],
    'Matemática': [90, 85, 78],
    'Ciências': [88, 92, 80],
    'História': [75, 78, 82]
})

# Utilizando a função melt para transformar o DataFrame largo em longo
df_longo = pd.melt(df_largo, id_vars=['Nome'], var_name='Matéria', value_name='Nota')

# Exibindo o DataFrame longo resultante
print(df_longo)

"""
id_vars: As colunas do DataFrame originais que você deseja manter fixas (sem derreter). No código fornecido, 'Nome' é uma coluna que queremos manter fixa, então é especificada como um valor para o parâmetro id_vars.

var_name: O nome da nova coluna que irá conter os nomes das colunas do DataFrame original (ou seja, as colunas que serão "derretidas"). No exemplo, as colunas 'Matemática', 'Ciências' e 'História' do DataFrame original serão "derretidas" em uma única coluna, que será chamada de 'Matéria'. Este é o propósito do parâmetro var_name.

value_name: O nome da nova coluna que conterá os valores que correspondem às colunas "derretidas". No exemplo, os valores das colunas 'Matemática', 'Ciências' e 'História' do DataFrame original serão "derretidos" em uma única coluna, que será chamada de 'Nota'. Este é o propósito do parâmetro value_name.
"""


"""
Pivot:
A função pivot é utilizada para reorganizar um DataFrame mais longo em um mais largo, criando uma tabela dinâmica.

"""

# Utilizando a função pivot para transformar o DataFrame longo de volta em largo
df_original = df_longo.pivot(index='Nome', columns='Matéria', values='Nota')

# Resetando o índice para tornar o DataFrame similar ao original
df_original.reset_index(inplace=True)

# Exibindo o DataFrame original resultante
print(df_original)
