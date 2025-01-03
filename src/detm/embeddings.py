from gensim.models import Word2Vec
import torch
import numpy


def train_embeddings(corpus, epochs=10, window_size=5, embedding_size=300, random_seed=None):
    model = Word2Vec(
        sentences=corpus.subdocs,
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
    return torch.tensor(numpy.array([embeddings.wv[w] for w in word_list]))
