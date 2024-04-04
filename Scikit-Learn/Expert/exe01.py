"""
Transfer Learning com Modelos Pré-treinados:

Utilize um modelo pré-treinado, como um modelo de linguagem BERT para tarefas específicas.
Adapte e ajuste o modelo para um conjunto de dados relacionado à sua área de interesse.


O Transfer Learning com modelos pré-treinados, como o BERT, é uma abordagem poderosa para tarefas específicas, pois esses modelos geralmente foram treinados em grandes quantidades de dados e capturaram conhecimento geral da linguagem. Vamos usar a biblioteca Transformers da Hugging Face para carregar e ajustar um modelo BERT para um conjunto de dados específico. 

"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulando um conjunto de dados fictício para classificação de sentimentos
data = {'text': ['Eu amo esse produto!', 'Não gostei da experiência.', 'Ótimo serviço!', 'Produto decepcionante.'],
        'sentiment': [1, 0, 1, 0]}
df = pd.DataFrame(data)

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Carregando o modelo BERT e o tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Criando uma classe Dataset personalizada para preparar os dados para o BERT
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Definindo hiperparâmetros
MAX_LEN = 20
BATCH_SIZE = 2
EPOCHS = 3

# Criando instâncias de DataLoader para treinamento e teste
train_dataset = CustomDataset(X_train, y_train, tokenizer, MAX_LEN)
test_dataset = CustomDataset(X_test, y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Configurando o otimizador e a função de perda
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
loss_fn = torch.nn.CrossEntropyLoss()

# Treinando o modelo
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(torch.long)
        attention_mask = batch['attention_mask'].to(torch.long)
        labels = batch['label'].to(torch.long)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

# Avaliando o modelo
model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(torch.long)
        attention_mask = batch['attention_mask'].to(torch.long)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        preds = np.argmax(logits.detach().numpy(), axis=1)
        all_preds.extend(preds)

# Avaliando a acurácia
accuracy = accuracy_score(y_test, all_preds)
print(f'Acurácia: {accuracy:.2f}')
