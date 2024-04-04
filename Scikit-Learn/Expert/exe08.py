"""
Processamento de Linguagem Natural (PNL) com Spacy:

Utilize a biblioteca spaCy para realizar tarefas avançadas de processamento de linguagem natural, como extração de entidades e análise sintática.

O processamento de linguagem natural (PNL) com spaCy é uma área ampla e poderosa. SpaCy é uma biblioteca Python de código aberto usada para realizar várias tarefas de PNL, incluindo tokenização, análise sintática, reconhecimento de entidades nomeadas (NER), lematização e muito mais.

"""

import spacy

# Carregar o modelo pré-treinado do spaCy para o idioma desejado
nlp = spacy.load("en_core_web_sm")

# Texto de exemplo para processamento
text = "Apple is looking at buying U.K. startup for $1 billion"

# Processar o texto com spaCy
doc = nlp(text)

# Tokenização
print("Tokens:")
for token in doc:
    print(token.text)

# Análise sintática
print("\nAnálise Sintática:")
for token in doc:
    print(token.text, token.dep_, token.head.text)

# Reconhecimento de Entidades Nomeadas (NER)
print("\nEntidades Nomeadas:")
for ent in doc.ents:
    print(ent.text, ent.label_)
