"""
Mesclagem de DataFrames:

Crie dois DataFrames e os mescle usando uma coluna em comum.

A mesclagem de DataFrames no Pandas pode ser realizada usando a função merge.

"""

import pandas as pd 


df1 = pd.DataFrame({
    'ID': [1, 2, 3],
    'Nome': ['Alice', 'Bob', 'Charlie'],
    'Cargo': ['Engenheiro', 'Analista', 'Gerente']
})

# criando o segundo DataFrame

import pandas as pd 


df2 = pd.DataFrame({
    'ID': [1, 2, 3],
    'Nome': ['Alice', 'Bob', 'Charlie'],
    'Cargo': ['Engenheiro', 'Analista', 'Gerente']
})


df_merged = pd.merge(df1, df2, on='ID', how='inner')

# exibindo o DataFrame mesclado
print(df_merged)


"""
O parâmetro how da função pd.merge() em Pandas pode receber os seguintes valores:

'inner': Realiza um merge interno (inner join), mantendo apenas as linhas onde a chave (coluna ou conjunto de colunas) está presente em ambos os DataFrames.

'outer': Realiza um merge externo (outer join), mantendo todas as linhas dos DataFrames originais e preenchendo os valores ausentes com NaN onde não há correspondência.

'left': Realiza um merge à esquerda (left join), mantendo todas as linhas do DataFrame à esquerda (df1) e preenchendo os valores ausentes com NaN onde não há correspondência no DataFrame à direita (df2).

'right': Realiza um merge à direita (right join), mantendo todas as linhas do DataFrame à direita (df2) e preenchendo os valores ausentes com NaN onde não há correspondência no DataFrame à esquerda (df1).

O how é um parâmetro opcional e seu valor padrão é 'inner'. Ele determina como o merge será realizado entre os DataFrames, e a escolha adequada depende do objetivo da operação de merge e da estrutura dos dados.




O parâmetro on na função pd.merge() especifica a(s) coluna(s) que será(ão) utilizada(s) como chave para realizar o merge entre os DataFrames. Este parâmetro é obrigatório e pode receber diferentes tipos de valores:

String ou Lista de Strings: Se apenas uma coluna de cada DataFrame for utilizada como chave, você pode passar o nome dessa coluna como uma string. Por exemplo, on='ID' indica que a coluna 'ID' será usada como chave para o merge.

Lista de Strings: Se você precisa fazer o merge com base em múltiplas colunas, pode passar uma lista contendo os nomes dessas colunas. Por exemplo, on=['ID', 'Nome'] indica que o merge será feito usando as colunas 'ID' e 'Nome' como chaves.

None (Padrão): Se on for definido como None, o Pandas irá realizar o merge com base nos índices dos DataFrames.


"""