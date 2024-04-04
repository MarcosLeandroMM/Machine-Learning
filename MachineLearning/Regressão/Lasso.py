"""
Assim como a regressão Ridge, a regressão Lasso adiciona um termo de regularização, mas utiliza a norma L1. A principal característica é que ela tende a gerar coeficientes esparsos, ou seja, alguns coeficientes podem ser exatamente zero.

"""

from sklearn.linear_model import Lasso

# Criando o modelo de regressão Lasso
modelo_lasso = Lasso(alpha=1.0)
modelo_lasso.fit(X, Y)

# Coeficientes
coeficientes_lasso = modelo_lasso.coef_
intercepcao_lasso = modelo_lasso.intercept_
