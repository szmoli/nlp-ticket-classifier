import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import word_tokenize
import string
import re
from gensim.models import FastText
import numpy as np

# Download nltk stuff if not downloaded already
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Converts treebank tag to wordnet pos tag
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Tokenizes and lemmatizes the given text
def clean_document(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\\[nrtbv]+', ' ', text)
    tokens = [token.lower().strip() for token in word_tokenize(text) if (token not in set(stopwords.words('english')) and token not in list(string.punctuation))]
    tagged_tokens = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(tagged_token[0], get_wordnet_pos(tagged_token[1])) for tagged_token in tagged_tokens]
    return lemmas

# Get vector for a document given by its tokens
def document_vector(tokens, model):
    word_vectors = []
    for token in tokens:
        if token in model.wv:
            word_vectors.append(model.wv[token])
        else:
            word_vectors.append(np.zeros(model.vector_size))
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

dataframe = pd.read_csv('data/dataset-tickets-multi-lang-5-2-50-version.csv')
dataframe_en = dataframe[dataframe['language'] == 'en'].copy()
clean_docs = [clean_document(body) for body in dataframe_en['body']]

print(f'Number of documents: {len(clean_docs)}')
print(f'Sample:\n{clean_docs[0]}')

ft_model = FastText(vector_size=100, window=5, min_count=1, workers=4, sg=1)
ft_model.build_vocab(clean_docs)
ft_model.train(clean_docs, total_examples=len(clean_docs), epochs=100)

doc_vectors = [document_vector(doc, ft_model) for doc in clean_docs]
print(f'Sample document vector:\n{doc_vectors[0]}')