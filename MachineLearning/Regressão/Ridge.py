"""
A regressão Ridge é uma extensão da regressão linear que adiciona um termo de regularização L2 à função de custo. Isso ajuda a evitar a multicolinearidade (alta correlação entre variáveis independentes) e a estabilizar os coeficientes. 


O parâmetro 
alpha (alfa) é um hiperparâmetro que controla a intensidade da regularização em modelos de regressão com regularização L1 (Lasso), L2 (Ridge) e Elastic Net. A escolha adequada de 
alpha é crucial para encontrar um equilíbrio entre ajustar bem os dados de treinamento e manter a simplicidade do modelo.

Parâmetro 
alpha (Regularização):

alpha na Regressão L1 (Lasso): Controla a força da penalização L1 nos coeficientes. Quanto maior o 
alpha
alpha, mais coeficientes se tornam exatamente zero, levando a um modelo mais esparsos.


alpha na Regressão L2 (Ridge): Controla a força da penalização L2 nos coeficientes. Quanto maior o 

alpha, mais os coeficientes são regularizados, diminuindo sua magnitude.


alpha na Regressão Elastic Net: É uma combinação ponderada de 

alpha para a regularização L1 e L2. Você ajusta 

alpha para controlar a força total da regularização, e 
l1_ratio
l1_ratio (discutido abaixo) controla a proporção entre L1 e L2.

Parâmetro 
l1_ratio
l1_ratio (Elastic Net):
l1_ratio
l1_ratio é um hiperparâmetro específico da regressão Elastic Net.
Varia de 0 a 1.
Quando  l1_ratio = 0 a Elastic Net se torna equivalente à Regressão Ridge.
Quando l1_ratio = 1, a Elastic Net se torna equivalente à Regressão Lasso.
Valores intermediários (0 < l1_ratio < 1) correspondem a combinações lineares das penalizações L1 e L2.

Ajustando alpha e  l1_ratio

l1_ratio:
Um alpha pequeno permite que o modelo se aproxime mais dos dados de treinamento, mas pode levar a overfitting.
Um alpha grande restringe mais os coeficientes, reduzindo a complexidade do modelo, mas pode resultar em um ajuste insuficiente.
O 
l1_ratio
l1_ratio permite ajustar o equilíbrio entre L1 e L2 na Elastic Net.
"""

from sklearn.linear_model import Ridge

# Criando o modelo de regressão Ridge
modelo_ridge = Ridge(alpha=1.0)
modelo_ridge.fit(X, Y)

# Coeficientes
coeficientes_ridge = modelo_ridge.coef_
intercepcao_ridge = modelo_ridge.intercept_
