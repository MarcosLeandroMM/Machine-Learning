"""
Tratamento de Outliers:

Identifique e lide com outliers em colunas numéricas.
O tratamento de outliers é uma parte importante da preparação dos dados, pois outliers podem distorcer análises estatísticas e modelos.


"""

"""

Identificação de Outliers:
1. Visualização Gráfica:
Utilize gráficos boxplot para visualizar a distribuição dos dados e identificar possíveis outliers.
"""
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# Criando um DataFrame de exemplo
dados = {'Coluna_Numérica': [10, 15, 20, 25, 200, 30, 35, 40, 45]}
df = pd.DataFrame(dados)

# Boxplot para identificar outliers
sns.boxplot(x=df['Coluna_Numérica'])
plt.show()


"""
Estatísticas Descritivas:
Calcule estatísticas descritivas e identifique valores que estão significativamente distantes da média.

"""

# Calculando estatísticas descritivas
estatisticas_descritivas = df['Coluna_Numérica'].describe()

# Identificando outliers usando critérios estatísticos (por exemplo, baseado no desvio padrão)
limite_superior = estatisticas_descritivas['mean'] + 3 * estatisticas_descritivas['std']
outliers = df[df['Coluna_Numérica'] > limite_superior]

# Exibindo os outliers
print("Outliers:")
print(outliers)


"""
Tratamento de Outliers:
1. Remoção:
Remova os outliers do DataFrame.

"""
# Removendo outliers
df_sem_outliers = df[df['Coluna_Numérica'] <= limite_superior]


"""
2. Substituição:
Substitua os outliers por valores mais adequados, como a mediana.

"""
# Substituindo outliers pela mediana
df['Coluna_Numérica'] = np.where(df['Coluna_Numérica'] > limite_superior, df['Coluna_Numérica'].median(), df['Coluna_Numérica'])

