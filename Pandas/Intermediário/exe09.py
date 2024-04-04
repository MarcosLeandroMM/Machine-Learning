"""
Juntando DataFrames com Merge:

Realize uma junção (merge) entre dois DataFrames usando diferentes tipos de junção (inner, outer, left, right).
A junção (merge) de DataFrames no Pandas pode ser feita usando a função merge. 


"""

import pandas as pd

# Inner Join:
# Criando dois DataFrames de exemplo
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Nome': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [2, 3, 4], 'Salário': [60000, 80000, 100000]})

# Inner join usando a coluna 'ID' 
# Um inner join retorna as linhas onde há correspondência nas duas tabelas (DataFrames) com base na(s) coluna(s) de junção. Resultado: Apenas as linhas com chaves correspondentes em ambos os DataFrames são incluídas no resultado.
df_inner = pd.merge(df1, df2, on='ID', how='inner')

# Exibindo o DataFrame resultante
print("Inner Join:")
print(df_inner)


# Outer Join: Um outer join retorna todas as linhas quando há correspondência em pelo menos uma das tabelas (DataFrames) com base na(s) coluna(s) de junção.Resultado: Todas as linhas são incluídas no resultado, e os valores não correspondentes em uma tabela são preenchidos com NaN (valores ausentes).
# Outer join usando a coluna 'ID'
df_outer = pd.merge(df1, df2, on='ID', how='outer')

# Exibindo o DataFrame resultante
print("\nOuter Join:")
print(df_outer)


# Left Join:Um left join retorna todas as linhas do DataFrame da esquerda (o primeiro DataFrame na chamada de merge) e as linhas correspondentes do DataFrame da direita.Resultado: Todas as linhas do DataFrame da esquerda são incluídas, com valores correspondentes do DataFrame da direita. Se não houver correspondência, os valores da direita serão NaN.
# Left join usando a coluna 'ID'
df_left = pd.merge(df1, df2, on='ID', how='left')

# Exibindo o DataFrame resultante
print("\nLeft Join:")
print(df_left)


# Right Join:Um right join retorna todas as linhas do DataFrame da direita (o segundo DataFrame na chamada de merge) e as linhas correspondentes do DataFrame da esquerda.Resultado: Todas as linhas do DataFrame da direita são incluídas, com valores correspondentes do DataFrame da esquerda. Se não houver correspondência, os valores da esquerda serão NaN.
# Right join usando a coluna 'ID'
df_right = pd.merge(df1, df2, on='ID', how='right')

# Exibindo o DataFrame resultante
print("\nRight Join:")
print(df_right)
