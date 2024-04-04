"""
Manipulação de Datas:

Converta uma coluna de datas para o tipo de dado datetime.
Extraia informações específicas, como dia da semana ou mês, dessa coluna.

"""

import pandas as pd 

dados = {'Data': ['2022-01-01', '2022-02-15', '2022-03-10', '2022-04-05'],
         'Valor': [10, 20, 15, 25]}

df = pd.DataFrame(dados)

# Convertendo a coluna 'Data' para datetime
df['Data'] = pd.to_datetime(df['Data'])

# Extraindo informações específicas
df['Dia_da_Semana'] = df['Data'].dt.day_name()
df['Mês'] = df['Data'].dt.month_name()

print(df)