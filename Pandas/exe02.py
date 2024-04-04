"""
Seleção de Dados:

Selecione uma única coluna do DataFrame.
Selecione um subconjunto de linhas e colunas.
Filtragem de Dados:

"""
import pandas as pd

# Seleção de uma única coluna:

meu_arquivo = 'dados.csv'
df = pd.read_csv(meu_arquivo)

coluna_nome = df['Nome']
print(coluna_nome)


# Seleção de um subconjunto de linhas e colunas:
subset = df.loc[:4,['Nome']['Idade']]
print(subset)

# Filtragem de Dados:
filtro_idade = df[df['Idade'] > 25]

'''
No pandas, iloc e loc são atributos usados para acessar e modificar dados em um DataFrame. Aqui está uma explicação de cada um:

iloc:

iloc é usado para acessar os dados por índices numéricos. Isso significa que você especifica a linha e a coluna desejadas usando seus índices numéricos.
A sintaxe básica é dataframe.iloc[linhas, colunas], onde linhas e colunas podem ser índices únicos, listas de índices ou fatias.
Por exemplo, df.iloc[0, 1] retorna o valor na primeira linha e segunda coluna do DataFrame df.
loc:

loc é usado para acessar os dados por rótulos (labels) de linha e/ou coluna. Isso significa que você especifica os nomes das linhas e/ou colunas em vez de seus índices numéricos.
A sintaxe básica é dataframe.loc[linhas, colunas], onde linhas e colunas podem ser rótulos únicos, listas de rótulos ou fatias.
Por exemplo, df.loc['A', 'B'] retorna o valor na linha 'A' e coluna 'B' do DataFrame df.
A diferença principal entre iloc e loc está nos métodos de indexação utilizados: iloc usa índices numéricos e loc usa rótulos. É importante entender o tipo de índices (numéricos ou rótulos) que seu DataFrame possui para escolher o método correto de acesso aos dados.
'''