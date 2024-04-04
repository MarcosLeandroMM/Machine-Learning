"""
Implementar uma Rede Neural Convolucional (CNN):
Construa uma CNN usando a biblioteca PyTorch.
Utilize módulos como nn.Conv2d, nn.MaxPool2d e nn.Linear.
Treine a CNN em um conjunto de dados de imagens.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Definir a arquitetura da CNN
class MinhaCNN(nn.Module):
    def __init__(self):
        super(MinhaCNN, self).__init__()
        self.camada_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.camada_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.camada_fc1 = nn.Linear(32 * 16 * 16, 256)
        self.camada_fc2 = nn.Linear(256, 10)  # 10 classes para ilustração

    def forward(self, x):
        x = self.relu(self.camada_conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.camada_conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(self.camada_fc1(x))
        x = self.camada_fc2(x)
        return x

# Hiperparâmetros
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# Transformações para o conjunto de dados (exemplo com CIFAR-10)
transformacoes = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Baixar conjunto de dados CIFAR-10 (para este exemplo)
conjunto_treinamento = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transformacoes)
loader_treinamento = torch.utils.data.DataLoader(conjunto_treinamento, batch_size=batch_size, shuffle=True)

# Instanciar a CNN
modelo_cnn = MinhaCNN()

# Definir função de perda e otimizador
criterio_cnn = nn.CrossEntropyLoss()
otimizador_cnn = optim.Adam(modelo_cnn.parameters(), lr=learning_rate)

# Treinar a CNN
for epoch in range(num_epochs):
    for imagens, rotulos in loader_treinamento:
        # Forward pass
        saidas_cnn = modelo_cnn(imagens)
        perda_cnn = criterio_cnn(saidas_cnn, rotulos)

        # Backward pass e otimização
        otimizador_cnn.zero_grad()
        perda_cnn.backward()
        otimizador_cnn.step()

    print(f'Época [{epoch+1}/{num_epochs}], Perda: {perda_cnn.item():.4f}')

print('Treinamento concluído!')

# Salvar o modelo treinado (opcional)
torch.save(modelo_cnn.state_dict(), 'modelo_cnn.pth')

# Agora você pode usar a CNN treinada para fazer previsões em novas imagens.

"""
Este código implementa uma CNN simples usando as camadas de convolução, ativação ReLU, max pooling e camadas totalmente conectadas. O conjunto de dados utilizado é o CIFAR-10, mas você pode adaptar o código para o seu próprio conjunto de dados. Certifique-se de ajustar os hiperparâmetros conforme necessário. O modelo treinado é salvo opcionalmente após o treinamento.

"""