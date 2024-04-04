"""
Implementar uma Rede Neural Recorrente (RNN):
Construa uma RNN utilizando a API tf.keras.layers.
Treine a RNN em dados sequenciais, como séries temporais ou texto.
Estes exercícios proporcionarão uma variedade de experiências com TensorFlow, desde tarefas básicas até conceitos mais avançados, como CNNs, Transfer Learning e treinamento distribuído. Lembre-se de adaptar os detalhes conforme necessário para atender às suas necessidades e interesses específicos.

Neste exemplo, usaremos a série temporal do conjunto de dados do Google Trends sobre o interesse pela palavra-chave "tensorflow.

"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Dados de séries temporais (interesse pela palavra-chave "tensorflow" no Google Trends)
# Os dados foram normalizados entre 0 e 1
series_temporais = np.array([0.02, 0.03, 0.03, 0.04, 0.07, 0.12, 0.17, 0.25, 0.45, 0.70, 
                             1.00, 0.70, 0.45, 0.25, 0.17, 0.12, 0.07, 0.04, 0.03, 0.03, 0.02])

# Função para criar sequências de entrada e saída
def criar_sequencias_dados(serie_temporal, tamanho_janela):
    sequencias = []
    proximo_valor = []
    for i in range(len(serie_temporal) - tamanho_janela):
        sequencia = serie_temporal[i:i+tamanho_janela]
        sequencias.append(sequencia)
        proximo_valor.append(serie_temporal[i+tamanho_janela])
    return np.array(sequencias), np.array(proximo_valor)

# Hiperparâmetros
tamanho_janela = 5
batch_size = 1

# Criar sequências de entrada e saída
X, y = criar_sequencias_dados(series_temporais, tamanho_janela)

# Expandir as dimensões dos dados de entrada para se adequar à entrada da RNN
X = np.expand_dims(X, axis=-1)

# Construir a RNN
modelo = models.Sequential([
    layers.SimpleRNN(20, input_shape=[None, 1]),
    layers.Dense(1)
])

# Compilar o modelo
modelo.compile(optimizer='adam', loss='mse')

# Treinar o modelo
history = modelo.fit(X, y, epochs=100)

# Plotar a perda durante o treinamento
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""
Importamos TensorFlow e outras bibliotecas relevantes.
Definimos uma série temporal que representa o interesse pela palavra-chave "tensorflow" no Google Trends.
Criamos sequências de entrada e saída a partir da série temporal.
Construímos uma RNN usando layers.SimpleRNN.
Compilamos o modelo com otimizador 'adam' e função de perda 'mse' (Mean Squared Error).
Treinamos o modelo com os dados de entrada e saída.
Plotamos a perda durante o treinamento para verificar o progresso do modelo.


"""
