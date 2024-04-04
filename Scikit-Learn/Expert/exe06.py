"""
Interpretabilidade do Modelo:

Utilize técnicas como LIME (Local Interpretable Model-agnostic Explanations) para entender e interpretar as decisões do modelo.

LIME (Local Interpretable Model-agnostic Explanations) é uma técnica usada para explicar as decisões de modelos de machine learning em nível local, ou seja, para instâncias de dados individuais. Ele fornece interpretabilidade para modelos de aprendizado de máquina, independentemente do algoritmo subjacente, tornando-se "agnostic" ao modelo.



"""

import lime
import lime.lime_tabular
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Carregar o conjunto de dados Iris
data = load_iris()
X = data.data
y = data.target

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o classificador (por exemplo, RandomForestClassifier)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Inicializar o explicador LIME
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode="classification", feature_names=data.feature_names, class_names=data.target_names)

# Selecionar uma instância de teste para explicação
instance_idx = 0
instance = X_test[instance_idx]
true_class = y_test[instance_idx]

# Obter a explicação do modelo para a instância selecionada
explanation = explainer.explain_instance(instance, classifier.predict_proba, num_features=len(data.feature_names), top_labels=1)

# Exibir a explicação
print("True Class:", data.target_names[true_class])
print("\nExplanation for the Top Class:")
explanation.show_in_notebook()
