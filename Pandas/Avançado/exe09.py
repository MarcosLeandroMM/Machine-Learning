"""
Análise de Sentimento Avançada:

Aplique técnicas avançadas de análise de sentimento em dados de texto usando bibliotecas como TextBlob ou NLTK.

"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize

# Baixe os recursos necessários (pode precisar apenas ser feito uma vez)
nltk.download('punkt')
nltk.download('vader_lexicon')

# Texto de exemplo
texto = "Eu amo este produto! É incrível, mas o serviço ao cliente é terrível."

# Tokenização de frases e palavras
frases = sent_tokenize(texto)
palavras = word_tokenize(texto)

# Análise de sentimento com NLTK SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
polaridade = sia.polarity_scores(texto)

# Exibindo resultados
print("Texto Original:", texto)
print("\nFrases Tokenizadas:", frases)
print("Palavras Tokenizadas:", palavras)
print("\nPolaridade do Sentimento:", polaridade)


"""
sent_tokenize é usado para dividir o texto em frases.
word_tokenize é usado para dividir o texto em palavras.
SentimentIntensityAnalyzer é uma classe do NLTK que fornece uma análise de sentimento mais avançada, incluindo polaridade.

"""