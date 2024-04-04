"""
Manipulação Avançada de Strings:

Utilize expressões regulares para extrair informações complexas de uma coluna de texto.
"""

import pandas as pd

# Criando um DataFrame de exemplo
dados = {'Texto': ['(11) 9876-5432', '(22) 5555-1234', '(33) 8765-4321']}
df = pd.DataFrame(dados)

# Aplicando regex para extrair números de telefone
df['Numero_Telefone'] = df['Texto'].str.extract(r'\((\d{2})\)\s*(\d{4,5}-\d{4})')

# Exibindo o DataFrame resultante
print(df)


"""
r'\((\d{2})\)\s*(\d{4,5}-\d{4})' é a expressão regular utilizada.

\((\d{2})\) corresponde a um parêntese de abertura, seguido por dois dígitos (código de área), seguido por um parêntese de fechamento.
\s* corresponde a zero ou mais espaços em branco.
(\d{4,5}-\d{4}) corresponde a quatro ou cinco dígitos, um hífen, e mais quatro dígitos (número de telefone).
df['Texto'].str.extract() é usado para aplicar a expressão regular à coluna 'Texto' e extrair os padrões correspondentes.

O resultado é uma nova coluna 'Numero_Telefone' no DataFrame, contendo apenas os números de telefone extraídos.
"""