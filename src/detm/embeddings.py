from gensim.models import Word2Vec
import numpy
    
def load_embeddings(fname):
    w2v = Word2Vec.load(fname)
    return w2v
    
def train_embeddings(corpus, content_field, max_subdoc_length, lowercase=True, epochs=10, window_size=5, embedding_size=300, random_seed=None):
    subdocs = corpus.get_tokenized_subdocs(
        max_subdoc_length=max_subdoc_length,
        content_field=content_field,
        lowercase=lowercase
    )
    model = Word2Vec(
        sentences=subdocs,
        vector_size=embedding_size,
        window=window_size,
        min_count=1,
        workers=4,
        sg=1,
        epochs=epochs,
        seed=random_seed
    )
    return model

def save_embeddings(embeddings, fname):
    embeddings.save(fname)
