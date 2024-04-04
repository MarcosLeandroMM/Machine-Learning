"""
Trabalhando com Dados Temporais Avançados:

Resample dados temporais para uma frequência diferente.
Realize análises de séries temporais avançadas.

"""

"""
Resample para uma Frequência Diferente:
Suponha que você tenha um DataFrame com dados temporais diários e deseje resample para uma frequência mensal

"""

import pandas as pd
import numpy as np

# Criando um DataFrame de exemplo com dados temporais diários
rng = pd.date_range('2022-01-01', '2022-12-31', freq='D')
df = pd.DataFrame({'Data': rng, 'Valor': np.random.randn(len(rng))})

# Resample para frequência mensal, usando a média dos valores mensais
df_resample = df.resample('M', on='Data').mean()

# Exibindo o DataFrame resample
print(df_resample)

"""
Neste exemplo, resample('M', on='Data').mean() resample os dados para uma frequência mensal e calcula a média dos valores mensais.
"""

"""
Análises de Séries Temporais Avançadas:
Para realizar análises mais avançadas, você pode usar técnicas como decomposição, autocorrelação, modelos ARIMA, entre outros. Vamos considerar um exemplo de decomposição:

"""

from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Adicionando uma tendência e sazonalidade ao DataFrame de exemplo
df['Tendencia'] = np.linspace(0, 10, len(df))
df['Sazonalidade'] = np.sin(2 * np.pi * df['Data'].dt.month / 12)

# Criando uma série temporal composta por tendência, sazonalidade e ruído
df['Serie_Temporal'] = df['Tendencia'] + df['Sazonalidade'] + np.random.randn(len(df))

# Decomposição da série temporal
resultados_decomposicao = seasonal_decompose(df['Serie_Temporal'], model='additive', period=12)

# Exibindo os componentes da decomposição
resultados_decomposicao.plot()
plt.show()

"""
Este exemplo utiliza a biblioteca statsmodels para realizar uma decomposição aditiva da série temporal em tendência, sazonalidade e ruído.

"""
