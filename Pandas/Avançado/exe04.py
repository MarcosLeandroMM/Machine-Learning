"""
Rolling Windows:

Utilize janelas deslizantes (rolling windows) para calcular estatísticas em movimento em uma série temporal.

As janelas deslizantes (rolling windows) no Pandas permitem calcular estatísticas em movimento em uma série temporal. Isso é útil para suavizar dados, identificar tendências e realizar análises de séries temporais. 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Criando uma série temporal de exemplo
rng = pd.date_range('2022-01-01', '2022-12-31', freq='D')
serie_temporal = pd.Series(np.random.randn(len(rng)), index=rng)

# Calculando a média móvel com uma janela deslizante de 7 dias
media_movel = serie_temporal.rolling(window=7).mean()

# Plotando a série temporal original e a média móvel
plt.figure(figsize=(10, 6))
plt.plot(serie_temporal, label='Série Temporal Original')
plt.plot(media_movel, label='Média Móvel (Janela de 7 dias)')
plt.legend()
plt.title('Série Temporal com Média Móvel')
plt.show()


"""Neste exemplo:

serie_temporal.rolling(window=7).mean() calcula a média móvel usando uma janela deslizante de 7 dias.
A função plot é usada para visualizar a série temporal original e a média móvel."""