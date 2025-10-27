from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download nltk stuff if not downloaded already
# try:
#     nltk.download('wordnet', quiet=True)
#     nltk.download('averaged_perceptron_tagger_eng', quiet=True)
#     nltk.download('omw-1.4', quiet=True)
#     nltk.download('punkt_tab', quiet=True)
#     nltk.download('stopwords', quiet=True)
# except:
#     pass

def get_wordnet_pos(tag):
    if tag.startswith('J'):  
        return 'a'
    elif tag.startswith('V'):  
        return 'v'
    elif tag.startswith('N'):  
        return 'n'
    elif tag.startswith('R'):  
        return 'r'
    else:
        return 'n'  

def tokenize(text):
    return [token.lower() for token in word_tokenize(text)]

def remove_stopwords(tokens, language = 'english'):
    return [token for token in tokens if (token not in set(stopwords.words(language)) and token not in list(string.punctuation))]

def lemmatize(tokens, lemmatizer = WordNetLemmatizer()):
    return [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for (token, tag) in pos_tag(tokens)]

# text = 'Dear Customer Support Team,\n\nI am writing to report a significant problem with the centralized account management portal, which currently appears to be offline. This outage is blocking access to account settings, leading to substantial inconvenience. I have attempted to log in multiple times using different browsers and devices, but the issue persists.\n\nCould you please provide an update on the outage status and an estimated time for resolution? Also, are there any alternative ways to access and manage my account during this downtime?'
# tokens = tokenize(text)
# useful_tokens = remove_stopwords(tokens)
# lemmas = lemmatize(useful_tokens)
# 
# print(text)
# print(tokens)
# print(useful_tokens)
# print(lemmas)