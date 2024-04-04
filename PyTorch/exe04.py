"""
4. Salvar e Carregar um Modelo Treinado:
Salve o modelo treinado em um arquivo usando torch.save.
Carregue o modelo salvo e faça previsões em novos dados.

"""

import torch

# Função para salvar o modelo
def salvar_modelo(modelo, caminho):
    torch.save(modelo.state_dict(), caminho)
    print(f'Modelo salvo em {caminho}')

# Função para carregar o modelo
def carregar_modelo(modelo, caminho):
    modelo.load_state_dict(torch.load(caminho))
    modelo.eval()
    print(f'Modelo carregado de {caminho}')

# Caminho para salvar o modelo
caminho_do_modelo = 'modelo_treinado.pth'

# Treinar o modelo (como no exercício 3)

# Salvar o modelo treinado
salvar_modelo(modelo, caminho_do_modelo)

# Criar um novo modelo para carregar os pesos salvos
modelo_carregado = ModeloSimples(input_size, output_size)

# Carregar o modelo treinado
carregar_modelo(modelo_carregado, caminho_do_modelo)

# Avaliar o modelo carregado
modelo_carregado.eval()
with torch.no_grad():
    novos_dados = torch.randn(5, input_size)  # Dados fictícios para predição
    previsoes = modelo_carregado(novos_dados)
    print("Previsões com modelo carregado:")
    print(previsoes)
