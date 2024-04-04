"""
Implementar uma Rede Neural Convolucional (CNN):
Construa uma CNN usando o TensorFlow.
Utilize camadas convolucionais, de pooling e totalmente conectadas.
Treine a CNN em um conjunto de dados de imagens.

"""


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Carregar e preparar os dados
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Construir a CNN
modelo = models.Sequential()
modelo.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
modelo.add(layers.MaxPooling2D((2, 2)))
modelo.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelo.add(layers.MaxPooling2D((2, 2)))
modelo.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelo.add(layers.Flatten())
modelo.add(layers.Dense(64, activation='relu'))
modelo.add(layers.Dense(10))

# Compilar a CNN
modelo.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Visualizar a arquitetura do modelo
modelo.summary()

# Treinar a CNN
history = modelo.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Avaliar o modelo
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = modelo.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

plt.show()
