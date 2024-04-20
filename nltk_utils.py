import nltk
import numpy as np

#nltk.download("punkt")
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bow(tokenized_sequence, all_words):
    tokenized_sequence = [stem(w) for w in tokenized_sequence] 
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sequence:
            bag[idx] = 1.0
    
    return bag

# a = ["hi", "there", "bro"]
# b = ["hello" ,"buddy","," ,"hi", "there", "bro"]
# print(bow(a, b))

