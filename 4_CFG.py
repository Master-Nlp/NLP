import nltk
from nltk import CFG, ChartParser

# Define the CFG grammar
grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N | N
VP -> V NP | V
Det -> 'the' | 'a'
N -> 'dog' | 'cat'
V -> 'chased' | 'saw'
""")
print("CFG defined successfully!")

# Parse a sentence
parser = ChartParser(grammar)
sentence = ['the', 'dog', 'chased', 'a', 'cat']

print("Parsing sentence:", ' '.join(sentence))
for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()
