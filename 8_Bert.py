# Step 1: Install Required Libraries (Run this in WSL terminal if not installed)
# pip install transformers torch pandas numpy scikit-learn matplotlib

# Step 2: Import Necessary Libraries
import pandas as pd
import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Step 3: Load Dataset
df = pd.read_csv("Bert_dataset.csv")  # Ensure dataset has 'Text' and 'Label' columns
print("Dataset Loaded Successfully!")
print(df.head())

# Step 4: Clean Text Data
def clean_text(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
    return text.lower().strip()

df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Step 5: Load Pre-trained BERT Model and Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Step 6: Prepare Dataset and DataLoader
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Map sentiment labels to numeric values
label_mapping = {"Positive": 0, "Negative": 1, "Neutral": 2}
df['Label'] = df['Label'].map(label_mapping)

# Create dataset and dataloader
dataset = SentimentDataset(df['Cleaned_Text'].tolist(), df['Label'].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Step 7: Define Training Function
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = CrossEntropyLoss()

def train_model(model, dataloader, optimizer, criterion, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

# Step 8: Train the Model
print("Training the model...")
train_model(model, dataloader, optimizer, criterion, epochs=3)

# Step 9: Define Sentiment Prediction Function
def predict_sentiment(text, tokenizer, model):
    encoding = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=128
    )
    outputs = model(**encoding)
    label = torch.argmax(outputs.logits).item()
    sentiment = ["Positive", "Negative", "Neutral"]
    return sentiment[label]

# Step 10: Test the Model
test_text = "I am so happy with this purchase!"
print(f"Sentiment: {predict_sentiment(test_text, tokenizer, model)}")
