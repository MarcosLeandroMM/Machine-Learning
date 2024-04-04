"""
Manipulação de Dados JSON:

Carregue dados JSON em um DataFrame e extraia informações específicas.
"""

import pandas as pd

# Carregando dados JSON em um DataFrame
df = pd.read_json('dados.json')

# Exibindo o DataFrame
print(df)

# Extração de Informações Específicas:
# Calculando a média da idade
media_idade = df['Idade'].mean()

# Exibindo a média da idade
print("Média da Idade:", media_idade)
