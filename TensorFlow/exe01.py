import tensorflow as tf

# Criar tensores
tensor1 = tf.constant([[1, 2], [3, 4]])  # Tensor 2x2
tensor2 = tf.constant([[5, 6], [7, 8]])  # Tensor 2x2

# Operações básicas
soma = tf.add(tensor1, tensor2)
subtracao = tf.subtract(tensor1, tensor2)
multiplicacao = tf.multiply(tensor1, tensor2)
divisao = tf.divide(tensor1, tensor2)

# Executar o grafo de operações
with tf.Session() as sess:
    resultado_soma = sess.run(soma)
    resultado_subtracao = sess.run(subtracao)
    resultado_multiplicacao = sess.run(multiplicacao)
    resultado_divisao = sess.run(divisao)

print("Tensor 1:")
print(tensor1)
print("Tensor 2:")
print(tensor2)

print("Soma:")
print(resultado_soma)
print("Subtração:")
print(resultado_subtracao)
print("Multiplicação:")
print(resultado_multiplicacao)
print("Divisão:")
print(resultado_divisao)

"""
Um tensor é uma estrutura de dados multidimensional, ou seja, é uma generalização de escalares, vetores e matrizes para dimensões superiores. Em termos mais simples, um tensor pode ser visto como um contêiner que pode armazenar dados em N dimensões.

Um tensor de ordem zero é um escalar, que é apenas um valor único.
Um tensor de primeira ordem é um vetor, que é uma lista ordenada de números.
Um tensor de segunda ordem é uma matriz, que é uma grade retangular de números.
Um tensor de ordem superior pode ser imaginado como uma estrutura multidimensional que contém matrizes aninhadas.
Em frameworks de aprendizado de máquina como TensorFlow, tensores são a principal estrutura de dados utilizada para representar dados de entrada, parâmetros do modelo, operações intermediárias e saídas do modelo durante o treinamento e inferência. Eles são fundamentais para expressar operações matemáticas eficientemente em redes neurais e outras técnicas de modelagem de dados.

"""