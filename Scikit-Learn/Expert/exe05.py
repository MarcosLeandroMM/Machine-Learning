"""
Classificação Multi-rótulo:

Aborde problemas de classificação multi-rótulo, onde uma instância pode pertencer a várias classes simultaneamente.


A classificação multi-rótulo é uma tarefa de aprendizado de máquina onde uma instância de dados pode pertencer a múltiplas classes ao mesmo tempo. Por exemplo, em um problema de classificação de imagens, uma única imagem pode conter várias entidades ou objetos diferentes, e queremos identificar todas essas entidades presentes na imagem.

Uma abordagem comum para lidar com problemas de classificação multi-rótulo é usar algoritmos que suportam essa funcionalidade diretamente, como o classificador MultiOutputClassifier do scikit-learn.


"""


from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Gerar dados de exemplo para classificação multi-rótulo
X, y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=5, n_labels=3, random_state=42)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o classificador base (por exemplo, Random Forest)
base_classifier = RandomForestClassifier(random_state=42)

# Inicializar o classificador MultiOutputClassifier
multi_output_classifier = MultiOutputClassifier(base_classifier, n_jobs=-1)

# Treinar o classificador
multi_output_classifier.fit(X_train, y_train)

# Fazer previsões
y_pred = multi_output_classifier.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Exibir relatório de classificação
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
