"""
Treinamento Distribuído com TensorFlow:
Configure um ambiente de treinamento distribuído usando TensorFlow, por exemplo, usando tf.distribute.MirroredStrategy.
Treine um modelo em vários dispositivos ou máquinas.

"""

import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import os

# Definir a estratégia de distribuição
strategy = tf.distribute.MirroredStrategy()

# Criar o escopo de distribuição
with strategy.scope():
    # Construir o modelo dentro do escopo de distribuição
    modelo = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compilar o modelo
    modelo.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Carregar e preparar os dados
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Treinar o modelo
modelo.fit(train_images, train_labels, epochs=5)

# Avaliar o modelo
test_loss, test_acc = modelo.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)


"""
Importamos TensorFlow e outras bibliotecas relevantes.
Definimos a estratégia de distribuição como tf.distribute.MirroredStrategy().
Criamos o escopo de distribuição usando o strategy.scope(), dentro do qual construímos o modelo. Isso garante que todas as operações relacionadas ao modelo estejam distribuídas entre os dispositivos disponíveis.
Compilamos o modelo dentro do escopo de distribuição.
Carregamos e preparamos os dados MNIST.
Treinamos o modelo usando modelo.fit().
Avaliamos o modelo usando modelo.evaluate().
Com a tf.distribute.MirroredStrategy, TensorFlow se encarregará de distribuir automaticamente o treinamento em várias GPUs disponíveis em uma única máquina. Se você tiver várias máquinas, pode configurar um cluster TensorFlow e usar a mesma estratégia para distribuir o treinamento em várias GPUs em várias máquinas.

"""