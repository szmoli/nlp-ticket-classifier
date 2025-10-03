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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import hashlib
import os
import pickle

# Download nltk stuff if not downloaded already
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

def file_hash(file):
    with open(file, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def cache_exists(cache_name, ext='.pkl'):
    return os.path.exists(os.path.join(CACHE_DIR, f"{cache_name}{ext}"))

def save_cache(data, cache_name):
    with open(os.path.join(CACHE_DIR, f"{cache_name}.pkl"), 'wb') as f:
        pickle.dump(data, f)

def load_cache(cache_name):
    with open(os.path.join(CACHE_DIR, f"{cache_name}.pkl"), 'rb') as f:
        return pickle.load(f)

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

CACHE_DIR = 'cache'
if not os.path.isdir(CACHE_DIR):
    os.mkdir(CACHE_DIR)

data_path = 'data/dataset-tickets-multi-lang-5-2-50-version.csv'
data_hash = file_hash(data_path)

dataframe = pd.read_csv(data_path)
dataframe_en = dataframe[dataframe['language'] == 'en'].copy()

# Clean documents
if cache_exists(f'clean_docs_{data_hash}'):
    print('Loading clean documents from cache...')
    clean_docs = load_cache(f'clean_docs_{data_hash}')
else:
    print('Cleaning documents...')
    clean_docs = [clean_document(body) for body in dataframe_en['body']]
    save_cache(clean_docs, f'clean_docs_{data_hash}')
    print(f'Clean documents cached')

print(f'Number of documents: {len(clean_docs)}')
print(f'Sample original:\n{dataframe_en["body"][1]}')
print(f'Sample clean:\n{clean_docs[0]}')

# Train model
if cache_exists(f'fasttext_model_{data_hash}', '.model'):
    print('Loading trained model from cache...')
    ft_model = FastText.load(os.path.join(CACHE_DIR, f'fasttext_model_{data_hash}.model'))
else:
    print('Training FastText model...')
    ft_model = FastText(vector_size=100, window=5, min_count=1, workers=4, sg=1)
    ft_model.build_vocab(clean_docs)
    ft_model.train(clean_docs, total_examples=len(clean_docs), epochs=100)
    ft_model.save(os.path.join(CACHE_DIR, f'fasttext_model_{data_hash}.model'))
    print('FastText model cached')

# Document vectors (= X)
if cache_exists(f'doc_vectors_{data_hash}'):
    print('Loading document vectors from cache...')
    doc_vectors = load_cache(f'doc_vectors_{data_hash}')
else:
    print('Calculating document vectors...')
    doc_vectors = [document_vector(doc, ft_model) for doc in clean_docs]
    save_cache(doc_vectors, f'doc_vectors_{data_hash}')
    print('Document vectors cached')

print(f'Sample document vector:\n{doc_vectors[0]}')

# y
categories = dataframe_en['queue'].values
print(f'Possible categories: {categories}')

X_train, X_test, y_train, y_test = train_test_split(doc_vectors, categories, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)
print(f'Accuracy: {clf.score(X_test, y_test):.3f}')