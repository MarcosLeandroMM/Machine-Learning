"""
1. Criar um Tensor e Realizar Operações Básicas:
Crie dois tensores simples em PyTorch.
Realize operações básicas de adição, subtração, multiplicação e divisão entre eles.

"""

import torch

# Criar tensores
tensor_a = torch.tensor([1.0, 2.0, 3.0])
tensor_b = torch.tensor([4.0, 5.0, 6.0])

# Operações básicas
soma = tensor_a + tensor_b
subtracao = tensor_a - tensor_b
multiplicacao = tensor_a * tensor_b
divisao = tensor_a / tensor_b

# Exibir resultados
print("Tensor A:", tensor_a)
print("Tensor B:", tensor_b)
print("Soma:", soma)
print("Subtração:", subtracao)
print("Multiplicação:", multiplicacao)
print("Divisão:", divisao)

"""
Um tensor é uma estrutura de dados fundamental em muitas bibliotecas de aprendizado de máquina e processamento de dados, como PyTorch, TensorFlow e NumPy. É uma generalização de matrizes para dimensões superiores. Vamos entender o conceito de tensor em diferentes dimensões:

Tensor 0D (Escalar): Um único número. Exemplo: 5.

Tensor 1D (Vetor): Uma sequência de números. Exemplo: [1, 2, 3].

Tensor 2D (Matriz): Uma tabela de números. Exemplo:


[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]

Tensor 3D: Uma coleção de matrizes. Pode ser imaginado como um cubo de números.

Tensor nD (onde n > 3): Pode ser uma generalização para tensores de ordens superiores.

No contexto de bibliotecas como PyTorch ou TensorFlow, tensores são utilizados para representar dados e operações em modelos de aprendizado de máquina. Eles são flexíveis e eficientes para realizar operações matemáticas em grandes conjuntos de dados.

Os tensores também podem ser utilizados em cálculos que envolvem gradientes, o que é crucial para algoritmos de aprendizado de máquina, pois permite o treinamento de modelos por meio de retropropagação (backpropagation), onde os gradientes são calculados para ajustar os pesos do modelo.

"""