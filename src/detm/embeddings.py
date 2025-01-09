import numpy, re
from gensim.models import Word2Vec

def train_embeddings(corpus, content_field, lowercase=True, 
                     epochs=10, window_size=5, embedding_size=300, random_seed=None):
    docs = [re.split(r"\s+", text[content_field].lower() if lowercase else text[content_field]) for text in corpus]
    
    model = Word2Vec(
        sentences=docs,
        vector_size=embedding_size,
        window=window_size,
        min_count=1,
        workers=4,
        sg=1,
        epochs=epochs,
        seed=random_seed
    )
    return model

def load_embeddings(fname):
    return Word2Vec.load(fname)

def save_embeddings(embeddings, fname):
    embeddings.save(fname)

def filter_embeddings(embeddings, word_list):
    return numpy.array([embeddings.wv[w] for w in word_list])