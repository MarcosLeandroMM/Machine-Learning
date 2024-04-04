"""
Visualização de Dados com Pandas:

Crie gráficos simples, como histogramas ou gráficos de dispersão, usando funções de visualização do Pandas.

"""

# Histograma
import pandas as pd
import matplotlib.pyplot as plt 

dados = {'Idade': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
         'Salário': [50000, 60000, 75000, 80000, 90000,
                      95000, 110000, 120000, 130000, 140000]}

df = pd.DataFrame(dados)

df['Idade'].plot(kind='hist', bins=5, edgecolor='black')
plt.title('Histograma de Idade')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

# Gráfico de Dispersão
# plotando um gráfico de dispersão entre 'Idade e 'Salário'