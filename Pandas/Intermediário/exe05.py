"""
Trabalhando com Strings:

Aplique operações de string em uma coluna, como dividir ou substituir caracteres.
"""

# Dividir uma Coluna de Strings
import pandas as pd

# Criando um DataFrame de exemplo
dados = {'Nome_Completo': ['Alice Smith', 'Bob Johnson', 'Charlie Brown']}
df = pd.DataFrame(dados)

# Dividindo a coluna 'Nome_Completo' em duas colunas: 'Primeiro_Nome' e 'Sobrenome'
df[['Primeiro_Nome', 'Sobrenome']] = df['Nome_Completo'].str.split(expand=True)

# Exibindo o DataFrame resultante
print(df)



# Substituir Caracteres em uma Coluna de Strings:
# Substituindo espaços em branco por underscores na coluna 'Nome_Completo'
df['Nome_Completo'] = df['Nome_Completo'].str.replace(' ', '_')

# Exibindo o DataFrame resultante
print(df)


# Extrair Parte de uma String:
# Extraindo os dois primeiros caracteres da coluna 'Primeiro_Nome'
df['Dois_Primeiros_Caracteres'] = df['Primeiro_Nome'].str[:2]

# Exibindo o DataFrame resultante
print(df)

