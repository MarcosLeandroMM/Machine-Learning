"""
Concatenação de DataFrames:

Crie dois DataFrames e concatene-os ao longo das linhas e colunas.
"""

import pandas as pd

# Criando dois DataFrames de exemplo
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2'],
    'C': ['C0', 'C1', 'C2']
})

df2 = pd.DataFrame({
    'A': ['A3', 'A4', 'A5'],
    'B': ['B3', 'B4', 'B5'],
    'C': ['C3', 'C4', 'C5']
})

# Concatenando ao longo das linhas (verticalmente)
df_concat_linhas = pd.concat([df1, df2], ignore_index=True)

# Concatenando ao longo das colunas (horizontalmente)
df_concat_colunas = pd.concat([df1, df2], axis=1)

# Exibindo os resultados
print("Concatenação ao longo das linhas:")
print(df_concat_linhas)

print("\nConcatenação ao longo das colunas:")
print(df_concat_colunas)
