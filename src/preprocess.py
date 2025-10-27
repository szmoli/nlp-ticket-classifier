from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer

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
    text = re.sub(r'\\[nrtbv]+', ' ', text)  
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text)
    return [token.lower() for token in word_tokenize(text)]

def get_domain_stopwords():
    return {
        'dear', 'team', 'hello', 'hi', 'please', 'thanks', 'thank', 'regards',
        'issue', 'problem', 'help', 'support', 'service', 'ticket', 'request',
        'hello', 'hi', 'hey', 'thanks', 'thank', 'please', 'regards', 'best',
        'kind', 'looking', 'forward', 'hello', 'hi', 'dear', 'sir', 'madam',
        'email', 'phone', 'call', 'contact', 'regarding', 'following'
    }

def remove_stopwords(tokens, language = 'english'):
    return [token for token in tokens if (token not in set(stopwords.words(language)) and token not in list(string.punctuation) and token not in get_domain_stopwords())]

def lemmatize(tokens, lemmatizer = WordNetLemmatizer()):
    return [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for (token, tag) in pos_tag(tokens)]