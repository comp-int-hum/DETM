from gensim.models.keyedvectors import KeyedVectors


def get_nearest_neighbors(model, words):
    kv = KeyedVectors(300, count=len(model.all_embeddings))
    kv.add_vectors([x for x, _ in model.all_embeddings], [x for _, x in model.all_embeddings])
    retval = {}
    for word in words:
        retval[word] = kv.most_similar(word)
    return retval


def get_topics(model):
    kv = KeyedVectors(300, count=len(model.all_embeddings))
    kv.add_vectors([x for x, _ in model.all_embeddings], [x for _, x in model.all_embeddings])
    alpha = model.get_alpha()[0].cpu().detach().numpy()
    kv.similar_by_vector(alpha[0,0,:])
    pass
