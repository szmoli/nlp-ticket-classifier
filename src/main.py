import pandas as pd
from gensim.models import FastText
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import preprocess
import cache
import vector

data_path = 'data/dataset-tickets-multi-lang-5-2-50-version.csv'
dataframe = pd.read_csv(data_path)
dataframe_en = dataframe[dataframe['language'] == 'en'].copy()
data_hash = cache.hash(data_path)

# Clean documents
clean_documents_path = f'cache/clean_documents_{data_hash}'
if cache.exists(clean_documents_path):
    print('Loading clean documents from cache...')
    clean_docs = cache.load(clean_documents_path)
else:
    print('Cleaning documents...')
    tokenized_documents = [preprocess.tokenize(body) for body in dataframe_en['body']]
    tokenized_documents = [preprocess.remove_stopwords(tokens) for tokens in tokenized_documents]
    clean_docs = [preprocess.lemmatize(tokens) for tokens in tokenized_documents]
    cache.save(clean_documents_path, clean_docs)
    print(f'Clean documents cached')

print(f'Number of documents: {len(clean_docs)}')
print(f'Sample original:\n{dataframe_en["body"][1]}')
print(f'Sample clean:\n{clean_docs[0]}')

# Train model
model_path = f'cache/model_{data_hash}'
if cache.exists(model_path):
    print('Loading trained model from cache...')
    ft_model = FastText.load(model_path)
else:
    print('Training FastText model...')
    ft_model = FastText(vector_size=150, window=7, min_count=2, workers=4, sg=1, epochs=50, sample=1e-5)
    ft_model.build_vocab(clean_docs)
    ft_model.train(clean_docs, total_examples=len(clean_docs), epochs=50)
    ft_model.save(model_path)
    print('FastText model cached')

# Document vectors (= X)
vectors_path = f'cache/vectors_{data_hash}'
if cache.exists(vectors_path):
    print('Loading document vectors from cache...')
    doc_vectors = load_cache(vectors_path)
else:
    print('Calculating document vectors...')
    doc_vectors = [vector.of(doc, ft_model) for doc in clean_docs]
    cache.save(vectors_path, doc_vectors)
    print('Document vectors cached')

print(f'Sample document vector:\n{doc_vectors[0]}')

# y
categories = dataframe_en['queue'].values
print(f'Possible categories: {categories}')

X_train, X_test, y_train, y_test = train_test_split(doc_vectors, categories, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)
print(f'Accuracy: {clf.score(X_test, y_test):.3f}')