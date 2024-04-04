"""
Otimização de Desempenho:

Otimize o desempenho do código Pandas, utilizando técnicas como apply vetorizado e cython.

"""

"""
Operações Vetorizadas:
Em vez de usar loops, tente utilizar operações vetorizadas do Pandas sempre que possível. Isso é mais eficiente.

"""

import pandas as pd
import numpy as np
import timeit

# Criando um DataFrame de exemplo
df = pd.DataFrame({'A': np.random.randn(1000), 'B': np.random.randn(1000)})

# Operação usando loop
start_time = timeit.default_timer()
df['C'] = df.apply(lambda row: row['A'] * row['B'], axis=1)
elapsed_time = timeit.default_timer() - start_time
print(f"Tempo com loop: {elapsed_time}")

# Operação vetorizada
start_time = timeit.default_timer()
df['C'] = df['A'] * df['B']
elapsed_time = timeit.default_timer() - start_time
print(f"Tempo vetorizado: {elapsed_time}")


"""Apply Vetorizado:
Quando o uso de apply é necessário, tente utilizar funções vetorizadas com numpy dentro da função aplicada."""
import pandas as pd
import numpy as np
import timeit

# Criando um DataFrame de exemplo
df = pd.DataFrame({'A': np.random.randn(1000), 'B': np.random.randn(1000)})

# Função aplicada usando apply
def minha_funcao(row):
    return row['A'] * row['B']

start_time = timeit.default_timer()
df['C'] = df.apply(minha_funcao, axis=1)
elapsed_time = timeit.default_timer() - start_time
print(f"Tempo com apply: {elapsed_time}")

# Função aplicada com numpy
start_time = timeit.default_timer()
df['C'] = np.vectorize(minha_funcao)(df['A'], df['B'])
elapsed_time = timeit.default_timer() - start_time
print(f"Tempo com apply vetorizado: {elapsed_time}")
