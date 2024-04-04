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


"""
Sequential é uma maneira de construir modelos de forma sequencial, camada por camada.
Dense define uma camada densa de neurônios, onde cada neurônio está conectado a todos os neurônios da camada anterior. No exemplo, temos uma camada densa com 2 neurônios e ativação ReLU (Rectified Linear Unit) seguida por uma camada densa com 1 neurônio e ativação sigmóide.
compile é usado para configurar as especificações de treinamento do modelo, incluindo o otimizador, a função de perda e as métricas a serem monitoradas durante o treinamento. Neste caso, usamos o otimizador Adam, a função de perda de entropia cruzada binária (adequada para problemas de classificação binária) e métrica de precisão.

 units e input_shape são argumentos usados na definição das camadas da rede neural.

units: Este parâmetro especifica o número de neurônios na camada. Cada neurônio em uma camada densa (ou totalmente conectada) está conectado a todos os neurônios da camada anterior. Portanto, o número de unidades controla o tamanho da representação aprendida pela camada. No exemplo dado, units=2 na primeira camada significa que haverá dois neurônios nessa camada.

input_shape: Este parâmetro especifica a forma dos dados de entrada que a camada espera. Ele é necessário apenas para a primeira camada da rede neural. No exemplo dado, input_shape=(2,) indica que os dados de entrada são vetores de comprimento 2. Isso é usado apenas na primeira camada para informar ao modelo o formato dos dados de entrada. Se você estiver lidando com imagens, por exemplo, input_shape seria definido como (largura, altura, canais).

"""