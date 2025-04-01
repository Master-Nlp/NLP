# Function to split text into individual characters
def char_tokenizer(text):
    return list(text)  # Return the list of characters

# Input text from the user
text = input("Enter a paragraph of text: ")

# Tokenize the text
tokens = char_tokenizer(text)

# Print the character tokens
print("Character Tokens:", tokens)
