"""A regressão Elastic Net é uma combinação da regressão Ridge e da regressão Lasso, utilizando termos de regularização L1 e L2. Isso ajuda a combinar as vantagens de ambas as abordagens."""
from sklearn.linear_model import ElasticNet

# Criando o modelo de regressão Elastic Net
modelo_elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)
modelo_elastic_net.fit(X, Y)

# Coeficientes
coeficientes_elastic_net = modelo_elastic_net.coef_
intercepcao_elastic_net = modelo_elastic_net.intercept_
