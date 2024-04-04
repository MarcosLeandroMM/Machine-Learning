"""
2. Construir um Modelo Simples:
Utilize a biblioteca PyTorch para construir um modelo de rede neural simples com uma ou duas camadas.
Defina uma função de perda e um otimizador.


"""

import torch
import torch.nn as nn
import torch.optim as optim

# Definir o modelo
class ModeloSimples(nn.Module):
    def __init__(self, input_size, output_size):
        super(ModeloSimples, self).__init__()
        self.camada_linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # A propagação para a frente (forward pass) da rede
        x = self.camada_linear(x)
        return x

# Parâmetros do modelo
input_size = 5  # Tamanho da entrada
output_size = 1  # Tamanho da saída

# Criar uma instância do modelo
modelo = ModeloSimples(input_size, output_size)

# Definir função de perda e otimizador
criterio = nn.MSELoss()  # Mean Squared Error Loss
otimizador = optim.SGD(modelo.parameters(), lr=0.01)  # Gradiente Descendente Estocástico

# Exibir a arquitetura do modelo
print(modelo)

"""
Neste exemplo, criamos um modelo simples chamado ModeloSimples com uma camada linear. Em seguida, definimos uma função de perda usando o erro médio quadrático (nn.MSELoss) e um otimizador usando o Gradiente Descendente Estocástico (optim.SGD). Certifique-se de ajustar os valores de input_size e output_size de acordo com os requisitos do seu problema. Este é um modelo básico e pode ser estendido conforme necessário para tarefas mais complexas.

"""