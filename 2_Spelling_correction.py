import re

def rule_based_pos_tagger(sentence):
    rules = {
        'NN': r'\b(the|a|an) [a-zA-Z]+\b',
        'ADV': r'\b\w+ly\b',
        'VBD': r'\b\w+ed\b',
    }
    
    words = sentence.split()
    tags = []

    for word in words:
        tag = 'UNK'
        for rule_tag, pattern in rules.items():
            if re.search(pattern, word, re.IGNORECASE):
                tag = rule_tag
        tags.append((word, tag))
    
    return tags

sentence = "The cat quickly jumped and walked."
print(rule_based_pos_tagger(sentence))
