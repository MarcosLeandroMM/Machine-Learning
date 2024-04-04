"""
Ordenação de Dados:

Ordene o DataFrame com base em uma ou mais colunas.

usar o método sort_values do Pandas

"""
import pandas as pd 

meu_arquivo = 'dados.csv'

df = pd.read_csv(meu_arquivo)


# Ordena o DataFrame com base na coluna 'Nome' em ordem ascendente
df_ordenado_nome = df.sort_values(by='Nome')

# Ordena o DataFrame com base nas colunas 'Idade' e 'Salário' em ordem descendente
df_ordenado_multiplas_colunas = df.sort_values(by=['Idade', 'Salário'], ascending=[False, False])

# O método sort_values retorna um novo DataFrame ordenado, para modificar o DataFrame original usar o 'inplace=True'

df.sort_values(by='Data', inplace=True)
