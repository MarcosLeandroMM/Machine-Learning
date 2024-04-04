"""
8. Implementar uma Rede Neural Recorrente (RNN):
Construa uma RNN utilizando módulos como nn.RNN ou nn.LSTM.
Treine a RNN em dados sequenciais, como séries temporais ou texto.

"""

import torch
import torch.nn as nn
import torch.optim as optim

# Definir a arquitetura da RNN
class MinhaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MinhaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out

# Hiperparâmetros
input_size = 10  # Tamanho da entrada
hidden_size = 20  # Tamanho do estado oculto
output_size = 1  # Tamanho da saída
num_epochs = 100
learning_rate = 0.01

# Gerar dados fictícios sequenciais
dados_sequenciais = torch.randn(100, 5, input_size)  # 100 sequências, cada uma com comprimento 5

# Rótulos fictícios (para ilustração)
rotulos = torch.randn(100, output_size)

# Instanciar a RNN
modelo_rnn = MinhaRNN(input_size, hidden_size, output_size)

# Definir função de perda e otimizador
criterio_rnn = nn.MSELoss()
otimizador_rnn = optim.SGD(modelo_rnn.parameters(), lr=learning_rate)

# Treinar a RNN
for epoch in range(num_epochs):
    saidas_rnn = modelo_rnn(dados_sequenciais)
    perda_rnn = criterio_rnn(saidas_rnn, rotulos)

    otimizador_rnn.zero_grad()
    perda_rnn.backward()
    otimizador_rnn.step()

    print(f'Época [{epoch+1}/{num_epochs}], Perda: {perda_rnn.item():.4f}')

print('Treinamento concluído!')

# Agora você pode usar a RNN treinada para fazer previsões em novos dados sequenciais.
