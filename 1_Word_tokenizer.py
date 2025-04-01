import re

def word_tokenizer(text):
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

text = input("Enter a paragraph of text: ")

tokens = word_tokenizer(text)
print("Word Tokens:", tokens)
