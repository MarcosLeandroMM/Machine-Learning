"""
Transfer Learning com TensorFlow:
Carregue um modelo pré-treinado, como o MobileNetV2 ou ResNet, usando tf.keras.applications.
Realize transfer learning ajustando o modelo para uma nova tarefa com um conjunto de dados diferente.


"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Carregar e preparar os dados
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Definir a base do modelo pré-treinado (MobileNetV2)
base_model = MobileNetV2(input_shape=(32, 32, 3),
                         include_top=False,
                         weights='imagenet')

# Congelar as camadas da base do modelo
base_model.trainable = False

# Adicionar camadas personalizadas no topo
modelo = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compilar o modelo
modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Visualizar a arquitetura do modelo
modelo.summary()

# Treinar o modelo
history = modelo.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Avaliar o modelo
test_loss, test_acc = modelo.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


"""
Carregamos os dados CIFAR-10 e normalizamos os valores dos pixels para o intervalo [0, 1].
Carregamos o modelo pré-treinado MobileNetV2, excluindo a camada de classificação final (include_top=False) e usando pesos pré-treinados na ImageNet.
Congelamos as camadas do MobileNetV2 para evitar que seus pesos sejam atualizados durante o treinamento.
Adicionamos camadas personalizadas no topo do modelo para adaptá-lo à tarefa de classificação de imagens do CIFAR-10.
Compilamos o modelo com otimizador 'adam' e função de perda 'sparse_categorical_crossentropy'.
Treinamos o modelo por 5 épocas usando os dados de treinamento e validação.
Avaliamos o desempenho do modelo nos dados de teste.


"""