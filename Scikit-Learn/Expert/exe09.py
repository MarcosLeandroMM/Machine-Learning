"""
Aprendizado Reforçado com Gym:
Integre o scikit-learn com a biblioteca OpenAI Gym para criar um agente de aprendizado reforçado em um ambiente simulado.

"""

import gym
import numpy as np
from sklearn.neural_network import MLPClassifier

# Criar o ambiente CartPole
env = gym.make('CartPole-v1')

# Defina uma função para converter observações em recursos:
def convert_observation_to_feature(observation):
    return np.array(observation).reshape(1, -1)


# Crie uma instância do classificador do scikit-learn (por exemplo, MLPClassifier) e inicialize-o
clf = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=1000)


# Execute o loop principal de treinamento e avaliação do agente:
for episode in range(100):
    # Reinicializar o ambiente para um novo episódio
    observation = env.reset()
    done = False
    
    # Loop de um único episódio
    while not done:
        # Renderizar o ambiente (opcional)
        env.render()
        
        # Converter a observação em recursos
        X = convert_observation_to_feature(observation)
        
        # Prever a ação com base nos recursos
        action = clf.predict(X)
        
        # Realizar a ação no ambiente
        observation, reward, done, info = env.step(action)
        
        # Treinar o classificador com a nova observação e a ação tomada
        clf.partial_fit(X, [action], classes=[0, 1])
