import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import word_tokenize
import string
import re

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
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\\[nrtbv]+', ' ', text)
    tokens = [token.lower().strip() for token in word_tokenize(text) if (token not in set(stopwords.words('english')) and token not in list(string.punctuation))]
    tagged_tokens = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(tagged_token[0], get_wordnet_pos(tagged_token[1])) for tagged_token in tagged_tokens]
    return lemmas

dataframe = pd.read_csv('data/dataset-tickets-multi-lang-5-2-50-version.csv')
dataframe_en = dataframe[dataframe['language'] == 'en'].copy()
clean_tokenized_texts = [clean_text(body) for body in dataframe_en['body']]
print(clean_tokenized_texts)
