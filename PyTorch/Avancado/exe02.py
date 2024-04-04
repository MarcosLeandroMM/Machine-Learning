"""
6. Transfer Learning com PyTorch:
Carregue um modelo pré-treinado, como o ResNet ou VGG, usando torchvision.models.
Realize transfer learning ajustando o modelo para uma nova tarefa com um conjunto de dados diferente.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# Hiperparâmetros
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# Transformações para o conjunto de dados CIFAR-10
transformacoes = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar as imagens para o tamanho esperado pela ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Baixar conjunto de dados CIFAR-10
conjunto_treinamento = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transformacoes)
loader_treinamento = torch.utils.data.DataLoader(conjunto_treinamento, batch_size=batch_size, shuffle=True)

# Carregar um modelo pré-treinado ResNet
modelo_resnet = models.resnet18(pretrained=True)

# Congelar os parâmetros da ResNet
for param in modelo_resnet.parameters():
    param.requires_grad = False

# Modificar a última camada totalmente conectada para se adequar à nova tarefa (CIFAR-10 tem 10 classes)
num_classes = 10
modelo_resnet.fc = nn.Linear(modelo_resnet.fc.in_features, num_classes)

# Definir função de perda e otimizador
criterio_resnet = nn.CrossEntropyLoss()
otimizador_resnet = optim.Adam(modelo_resnet.fc.parameters(), lr=learning_rate)

# Treinar o modelo
for epoch in range(num_epochs):
    for imagens, rotulos in loader_treinamento:
        # Forward pass
        saidas_resnet = modelo_resnet(imagens)
        perda_resnet = criterio_resnet(saidas_resnet, rotulos)

        # Backward pass e otimização
        otimizador_resnet.zero_grad()
        perda_resnet.backward()
        otimizador_resnet.step()

    print(f'Época [{epoch+1}/{num_epochs}], Perda: {perda_resnet.item():.4f}')

print('Transfer Learning concluído!')

"""
Este exemplo usa a ResNet-18 pré-treinada da torchvision, ajusta a última camada totalmente conectada para se adequar ao novo conjunto de dados (CIFAR-10) e treina o modelo resultante no conjunto de dados CIFAR-10. Lembre-se de que, para tarefas reais, você pode precisar ajustar ainda mais o modelo, como adicionar camadas adicionais ou ajustar hiperparâmetros.

"""