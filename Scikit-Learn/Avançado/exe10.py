"""
Análise de Sentimento com NLP:

Aplique técnicas de pré-processamento de texto e use um modelo de classificação para análise de sentimento.

Para realizar uma Análise de Sentimento com NLP (Processamento de Linguagem Natural), podemos usar técnicas de pré-processamento de texto e um modelo de classificação

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Carregando o conjunto de dados IMDB
# Certifique-se de ter o arquivo 'imdb_reviews.csv' no diretório correto
df = pd.read_csv('imdb_reviews.csv')

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Criando um pipeline com TfidfVectorizer e Random Forest Classifier
model = make_pipeline(TfidfVectorizer(), StandardScaler(with_mean=False), RandomForestClassifier(random_state=42))

# Treinando o modelo
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Exibindo métricas de classificação detalhadas
print('Métricas de Classificação:')
print(classification_report(y_test, y_pred))


"""
Carregamos o conjunto de dados IMDB, que contém avaliações de filmes e seus sentimentos associados.
Dividimos o conjunto de dados em conjuntos de treinamento e teste.
Criamos um pipeline que utiliza TfidfVectorizer para converter o texto em características vetoriais e um classificador RandomForestClassifier para realizar a classificação.
Treinamos o modelo usando o conjunto de treinamento.
Fazemos previsões no conjunto de teste.
Avaliamos o desempenho do modelo usando a acurácia e métricas de classificação detalhadas.

"""