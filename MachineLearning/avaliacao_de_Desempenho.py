from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Exemplo de avaliação de desempenho
Y_predito = modelo_arvore.predict(X_teste)

acuracia = accuracy_score(Y_teste, Y_predito)
precisao = precision_score(Y_teste, Y_predito, average='weighted')
revocacao = recall_score(Y_teste, Y_predito, average='weighted')
f1 = f1_score(Y_teste, Y_predito, average='weighted')
matriz_confusao = confusion_matrix(Y_teste, Y_predito)

print(f'Acurácia: {acuracia}')
print(f'Precisão: {precisao}')
print(f'Revocação: {revocacao}')
print(f'F1-Score: {f1}')
print(f'Matriz de Confusão:\n{matriz_confusao}')
