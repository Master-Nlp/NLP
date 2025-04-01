import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Define a sample sentence with a polysemous word
sentence = "The fishermen sat on the bank of the river."
word = "bank"

# Apply the Lesk Algorithm
sense = lesk(nltk.word_tokenize(sentence), word)

# Print results
if sense:
    print("Best Sense: ", sense)
    print("Definition: ", sense.definition())
else:
    print("No sense found.")
