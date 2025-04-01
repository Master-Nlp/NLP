# Step 1: Installation and Setup
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
nltk.download('punkt')

# Step 2: Prepare Data for Training
corpus = [
    "Word embeddings are helpful in natural language processing",
    "Word2Vec is a powerful technique for learning word embeddings",
    "Machine learning and deep learning techniques are used in NLP",
    "Word2Vec uses neural networks to learn vector representations"
]

# Tokenize the corpus
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]
print("Tokenized Corpus:", tokenized_corpus)

# Step 3: Train a Word2Vec Model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=3,
                  min_count=1, sg=1)

# Print the vocabulary
print("Vocabulary:", model.wv.index_to_key)

# Step 4: Explore Word Embeddings
# Get the vector representation of a word
try:
    word_vector = model.wv['word2vec']
    print("Word2Vec Vector for 'word2vec':", word_vector)
except KeyError:
    print("Word 'word2vec' not in vocabulary. Try another word.")

# Find the most similar words to a given word
try:
    similar_words = model.wv.most_similar('word')
    print("Most similar words to 'word':", similar_words)
except KeyError:
    print("Word 'word' not in vocabulary. Try another word.")

# Step 5: Save and Load the Model
model.save("word2vec.model")
print("Model saved successfully!")

# Load the saved model
loaded_model = Word2Vec.load("word2vec.model")
print("Model loaded successfully!")
