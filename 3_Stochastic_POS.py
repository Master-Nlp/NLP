import nltk
from nltk.tokenize import word_tokenize
from nltk import word_tokenize
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('universal_tagset')
# nltk.download('stopwords')

def stochastic_pos_tagger(sentence):
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens, tagset='universal')
    return pos_tags

sentence = "The cat quickly jumped and walked."
print(stochastic_pos_tagger(sentence))
