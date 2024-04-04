"""
Trabalhando com Multi-índices:

Crie um DataFrame com um índice hierárquico (multi-índice) e realize operações específicas.

Trabalhar com Multi-índices no Pandas permite criar estruturas de dados mais complexas, com índices hierárquicos.
"""

import pandas as pd

# Criando um DataFrame com Multi-índice
dados = {
    'Valor': [10, 20, 15, 25, 12, 18],
    'Quantidade': [50, 30, 40, 20, 45, 35]
}

indices = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1), ('B', 2), ('C', 1), ('C', 2)],
                                    names=['Grupo', 'Subgrupo'])

df = pd.DataFrame(dados, index=indices)

# Exibindo o DataFrame com Multi-índice
print("DataFrame com Multi-índice:")
print(df)

# Acessando dados por índices hierárquicos
valor_subgrupo_B2 = df.loc[('B', 2), 'Valor']
print("\nValor no Subgrupo B2:", valor_subgrupo_B2)

# Agregação por nível do índice hierárquico
soma_por_grupo = df.groupby('Grupo').sum()
print("\nSoma por Grupo:")
print(soma_por_grupo)

"""
Um Multi-índice é criado usando o método pd.MultiIndex.from_tuples().
O DataFrame é criado com esse Multi-índice.
Dados podem ser acessados usando as tuplas que compõem os índices hierárquicos.
Operações de agrupamento podem ser realizadas por níveis do índice hierárquico usando o método groupby.

"""
