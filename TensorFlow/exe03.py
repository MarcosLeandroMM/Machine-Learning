"""
Treinar o Modelo:
Gere dados de treinamento e treine o modelo construído no exercício 2.
Use a API de treinamento do TensorFlow para iterar sobre os dados e ajustar os pesos do modelo.

"""

import tensorflow as tf

# Definindo os dados de entrada e saída (exemplo fictício)
dados_entrada = [[0, 0], [0, 1], [1, 0], [1, 1]]
dados_saida = [0, 1, 1, 0]

# Construindo o modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compilando o modelo
modelo.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

# Visualizando a arquitetura do modelo
modelo.summary()


import numpy as np

# Gerando dados de treinamento fictícios
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0])

# Treinando o modelo
modelo.fit(X_train, y_train, epochs=1000, verbose=1)
