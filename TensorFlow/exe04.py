"""
Salvar e Carregar um Modelo Treinado:
Salve o modelo treinado em um arquivo.
Carregue o modelo salvo e faça previsões em novos dados.

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



# Supondo que `modelo` é o modelo treinado que você deseja salvar

# Salvando o modelo
modelo.save('modelo_treinado.h5')

# Carregando o modelo salvo
modelo_carregado = tf.keras.models.load_model('modelo_treinado.h5')

# Exemplo de previsão em novos dados usando o modelo carregado
novos_dados = np.array([[0, 1], [1, 0]])
previsoes = modelo_carregado.predict(novos_dados)
print("Previsões:")
print(previsoes)
