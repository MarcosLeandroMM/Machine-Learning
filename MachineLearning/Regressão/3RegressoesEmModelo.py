from sklearn.linear_model import Lasso, Ridge, ElasticNet

# Criando modelos com diferentes valores de alpha
modelo_lasso = Lasso(alpha=0.1)
modelo_ridge = Ridge(alpha=0.1)
modelo_elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Ajustando os modelos aos dados
modelo_lasso.fit(X, Y)
modelo_ridge.fit(X, Y)
modelo_elastic_net.fit(X, Y)

# Coeficientes resultantes
coef_lasso = modelo_lasso.coef_
coef_ridge = modelo_ridge.coef_
coef_elastic_net = modelo_elastic_net.coef_
