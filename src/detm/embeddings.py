from gensim.models import Word2Vec


def train_embeddings(corpus, epochs=10, window_size=5, embedding_size=300, random_seed=None):
    model = Word2Vec(
        sentences=corpus.all_subdocs(),
        vector_size=embedding_size,
        window=window_size,
        min_count=1,
        workers=4,
        sg=1,
        epochs=epochs,
        seed=random_seed
    )
    return model
