from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import gzip
    
def load_embeddings(fname):
    embeddings = KeyedVectors.load(fname)
    return embeddings
    
def train_embeddings(corpus, content_field, epochs=10, window_size=5, embedding_size=300, random_seed=None):
    subdocs = corpus.get_tokenized_subdocs(
        content_field=content_field,
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
    return model.wv

def save_embeddings(embeddings, fname):
    with gzip.open(fname, "wb") as ofd:
        embeddings.save(ofd, separately=[])
