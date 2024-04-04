"""
3. Treinar o Modelo:
Gere dados de treinamento e treine o modelo construído no exercício 2.
Use um loop de treinamento padrão para iterar sobre os dados e ajustar os pesos do modelo.


"""

import torch

# Função para gerar dados fictícios
def gerar_dados(tamanho_do_conjunto):
    # Vamos assumir um problema de regressão simples
    features = torch.randn(tamanho_do_conjunto, input_size)
    labels = 2 * features[:, 0] - 3 * features[:, 1] + 1.5 * features[:, 2] + torch.randn(tamanho_do_conjunto)
    labels = labels.view(-1, 1)  # Adaptação das dimensões para corresponder à saída do modelo
    return features, labels

# Hiperparâmetros
num_epochs = 1000
tamanho_do_conjunto = 100
taxa_de_aprendizado = 0.01

# Gerar dados de treinamento
dados_de_treinamento, alvos_de_treinamento = gerar_dados(tamanho_do_conjunto)

# Instanciar o modelo
modelo = ModeloSimples(input_size, output_size)

# Definir função de perda e otimizador
criterio = nn.MSELoss()
otimizador = optim.SGD(modelo.parameters(), lr=taxa_de_aprendizado)

# Loop de treinamento
for epoch in range(num_epochs):
    # Forward pass
    saidas_do_modelo = modelo(dados_de_treinamento)
    perda = criterio(saidas_do_modelo, alvos_de_treinamento)

    # Backward pass e otimização
    otimizador.zero_grad()
    perda.backward()
    otimizador.step()

    # Exibir a perda a cada 100 épocas
    if (epoch + 1) % 100 == 0:
        print(f'Época [{epoch+1}/{num_epochs}], Perda: {perda.item():.4f}')

# Avaliar o modelo treinado
modelo.eval()
with torch.no_grad():
    novos_dados = torch.randn(5, input_size)  # Dados fictícios para predição
    previsoes = modelo(novos_dados)
    print("Previsões:")
    print(previsoes)

"""
Este código ilustra um loop de treinamento simples usando dados fictícios. Na prática, você substituiria gerar_dados pelos seus próprios dados de treinamento. Durante cada iteração do loop, o modelo faz uma propagação para frente, calcula a perda, realiza a retropropagação e atualiza os pesos usando o otimizador.

Por fim, o modelo treinado é utilizado para fazer previsões em novos dados (novos_dados), e as previsões são exibidas. Lembre-se de que este é apenas um exemplo básico, e em problemas do mundo real, você precisaria ajustar os hiperparâmetros, lidar com conjuntos de dados maiores e talvez explorar arquiteturas de modelo mais complexas.

"""