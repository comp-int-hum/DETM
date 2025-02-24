from gensim.models import CoherenceModel
import gensim.downloader as api
import numpy as np
import numpy as np
from .rbo import rbo
from scipy.spatial import distance
from itertools import combinations
from .word_embeddings_rbo import word_embeddings_rbo
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import preprocess_string
import torch
import random


def random_pairs(n, k):
    """
    Randomly sample k distinct unordered pairs (i, j), i < j,
    from the range 0..n-1. 
    """
    pairs = set()
    while len(pairs) < k:
        i = random.randrange(n)
        j = random.randrange(n)
        if i != j:
            # Order them so (i,j) = (min, max)
            if j < i:
                i, j = j, i
            pairs.add((i, j))
    return list(pairs)


def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l]
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l]
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj 

def get_topic_coherence(beta, data):
    D = len(data) ## number of docs...data is list of documents
    print('D: ', D)
    TC = []
    num_topics = len(beta)
    for k in range(num_topics):
        #print('k: {}/{}'.format(k, num_topics))
        top_10 = list(beta[k].argsort()[-11:][::-1])
        #top_words = [vocab[a] for a in top_10]
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
                # update tmp: 
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp 
        TC.append(TC_k/counter)
    #print('counter: ', counter)
    #print('num topics: ', len(TC))
    TC = np.mean(TC)
    #print('Topic Coherence is: {}'.format(TC))
    return TC, counter

def _diversity_helper(beta, num_tops, model):
    list_w = np.zeros((model.num_topics, num_tops))
    for k in range(model.num_topics):
        gamma = beta[k, :]
        top_words = gamma.cpu().numpy().argsort()[-num_tops:][::-1]
        list_w[k, :] = top_words
    list_w = np.reshape(list_w, (-1))
    list_w = list(list_w)
    n_unique = len(np.unique(list_w))
    diversity = n_unique / (model.num_topics * num_tops)
    return diversity

def original_detm_evaluation(model, dataset):
    """Returns topic coherence and topic diversity.
    """
    model.eval()
    with torch.no_grad():
        beta = model.topic_distributions()
        beta = beta.transpose(0, 1)
        #print('beta: ', beta.size())

        #print('\n')
        #print('#'*100)
        #print('Get topic diversity...')
        num_tops = 25
        TD_all = [0] * model.num_windows
        for tt in range(model.num_windows):
            TD_all[tt] = _diversity_helper(beta[:, tt, :], num_tops, model)
        TD = np.mean(TD_all)
        #print('Topic Diversity is: {}'.format(TD))

        #print('\n')
        #print('Get topic coherence...')
        #print('train_tokens: ', train_tokens[0])
        TC_all = []
        cnt_all = []
        for tt in range(model.num_windows):
            tc, cnt = get_topic_coherence(beta[:, tt, :].cpu().numpy(), dataset)
            TC_all.append(tc.item())
            cnt_all.append(cnt)
        #print('TC_all: ', TC_all)
        TC_all = TC_all
        #print('TC_all: ', torch.tensor(TC_all).size())
        #print('\n')
        #print('Get topic quality...')
        quality = [td*tc for td, tc in zip(TD_all, TC_all)]
        #print('Topic Quality is: {}'.format(quality))
        #print('#'*100)
    return TD_all, TC_all, quality


def evaluate_coherence(model=None, topics=None, coherence_measure="c_v", topn=10, text=None, dictionary=None, **args):
    """
    Evaluate the coherence of the model

    Parameters
    ----------
    model : model
    coherence_measure : str, 'c_v', 'c_uci', 'c_npmi', 'u_mass'
    text : list of list of str (optional

    Returns
    -------
    coherence : float, dict mean coherence, coherence per window { window_index : coherence }
    """
    assert coherence_measure in ['c_v', 'c_uci', 'c_npmi', 'u_mass', 'c_w2v'], 'coherence measure not recognized'
    assert model is not None or topics is not None, 'model or topics must be provided'

    if text is None:
        corpus = api.load('20-newsgroups')
        text = [
            preprocess_string(text['data'])
            for text in corpus
        ]
    if not dictionary:
        dictionary = Dictionary(text)

    if not topics:
        topics = model.get_topic_words(topn)

    word_list = [topic for window in topics for topic in window]
    dictionary.add_documents(word_list)

    num_windows = len(topics)
    coherences = {}
    for window in range(num_windows):
        try:
            coherence_model = CoherenceModel(topics=topics[window], texts=text, coherence=coherence_measure, topn=topn, dictionary=dictionary, **args)
            coherences[window] = coherence_model.get_coherence().item()
        except Exception as e:
            print(f"Error in coherence calculation in window {window}")
            raise e
    return (np.mean(list(coherences.values())).item(), coherences)

def evaluate_topic_diversity(model, divergence_measure, topn=10, **args):
    """
    Evaluate the diversity of the model

    Parameters
    ----------
    model : model
    divergence_measure: 'proportion_unique_words', 'irbo', 'word_embedding_irbo', 'pairwise_jaccard_diversity', 'pairwise_word_embedding_distance', 'centroid_distance'
    topn : int

    Returns
    -------
    diversity : float
    """
    topics = model.get_topic_words(topn)
    num_windows = len(topics)
    diversities = {}
    for window in range(num_windows):
        topic_words = topics[window]
        if divergence_measure == 'proportion_unique_words':
            diversities[window] = proportion_unique_words(topic_words, topn)
        elif divergence_measure == 'irbo':
            diversities[window] = irbo(topic_words, topk=topn)
        elif divergence_measure == 'word_embedding_irbo':
            assert 'embedding' in args, 'word_embedding_irbo requires an embedding model'
            embedding = args['embedding']
            diversities[window] = word_embedding_irbo(topic_words, embedding, topk=topn)
        elif divergence_measure == 'pairwise_jaccard_diversity':
            diversities[window] = pairwise_jaccard_diversity(topic_words, topk=topn)
        elif divergence_measure == 'pairwise_word_embedding_distance':
            assert 'embedding' in args, 'pairwise_word_embedding_distance requires an embedding model'
            embedding = args['embedding']
            diversities[window] = pairwise_word_embedding_distance(topic_words, embedding, topk=topn)
        elif divergence_measure == 'centroid_distance':
            assert 'embedding' in args, 'centroid_distance requires an embedding model'
            embedding = args['embedding']
            diversities[window] = centroid_distance(topic_words, embedding, topk=topn)
        else:
            raise Exception('Divergence measure not recognized')
    return (np.mean(list(diversities.values())).item(), diversities)



# SOURCE Topic diversity code: https://github.com/silviatti/topic-model-diversity/tree/master


def proportion_unique_words(topics, topk=10):
    """
    compute the proportion of unique words

    Parameters
    ----------
    topics: a list of lists of words
    topk: top k words on which the topic diversity will be computed
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than '+str(topk))
    else:
        unique_words = set()
        for topic in topics:
            unique_words = unique_words.union(set(topic[:topk]))
        puw = len(unique_words) / (topk * len(topics))
        return puw


def irbo(topics, weight=0.9, topk=10, sample=10000):
    """
    compute the inverted rank-biased overlap

    Parameters
    ----------
    topics: a list of lists of words
    weight: p (float), default 1.0: Weight of each
        agreement at depth d:p**(d-1). When set
        to 1.0, there is no weight, the rbo returns
        to average overlap.
    topk: top k words on which the topic diversity
          will be computed
    sample: number of samples to take from the
            rank-biased overlap

    Returns
    -------
    irbo : score of the rank biased overlap over the topics
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        num_topics = len(topics)
        num_combinations = (num_topics * (num_topics-1))/2
        if num_combinations > sample:
            pair_idxs = random_pairs(num_topics, sample)
            pairs = [(topics[i], topics[j]) for i, j in pair_idxs]
        else:
            pairs = combinations(topics, 2)
        collect = []
        for list1, list2 in pairs:
            word2index = get_word2index(list1, list2)
            indexed_list1 = [word2index[word] for word in list1]
            indexed_list2 = [word2index[word] for word in list2]
            rbo_val = rbo(indexed_list1[:topk], indexed_list2[:topk], p=weight)[2]
            collect.append(rbo_val)
        return 1 - np.mean(collect)


def word_embedding_irbo(topics, word_embedding_model, weight=0.9, topk=10, sample=10000):
    '''
    compute the word embedding-based inverted rank-biased overlap

    Parameters
    ----------
    topics: a list of lists of words
    weight: p (float), default 1.0: Weight of each agreement at depth d:
    p**(d-1). When set to 1.0, there is no weight, the rbo returns to average overlap.
    
    Returns
    -------
    weirbo: word embedding-based inverted rank_biased_overlap over the topics
    '''
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        num_topics = len(topics)
        num_combinations = (num_topics * (num_topics-1))/2
        if num_combinations > sample:
            pair_idxs = random_pairs(num_topics, sample)
            pairs = [(topics[i], topics[j]) for i, j in pair_idxs]
        else:
            pairs = combinations(topics, 2)
        collect = []
        for list1, list2 in pairs:
            word2index = get_word2index(list1, list2)
            index2word = {v: k for k, v in word2index.items()}
            indexed_list1 = [word2index[word] for word in list1]
            indexed_list2 = [word2index[word] for word in list2]
            rbo_val = word_embeddings_rbo(indexed_list1[:topk], indexed_list2[:topk], p=weight,
                                          index2word=index2word, word2vec=word_embedding_model)[2]
            collect.append(rbo_val)
        return 1 - np.mean(collect)


def pairwise_jaccard_diversity(topics, topk=10, sample=10000):
    '''
    compute the average pairwise jaccard distance between the topics 
  
    Parameters
    ----------
    topics: a list of lists of words
    topk: top k words on which the topic diversity
          will be computed
    
    Returns
    -------
    pjd: average pairwise jaccard distance
    '''
    dist = 0
    count = 0
    num_topics = len(topics)
    num_combinations = (num_topics * (num_topics-1))/2
    if num_combinations > sample:
        pair_idxs = random_pairs(num_topics, sample)
        pairs = [(topics[i], topics[j]) for i, j in pair_idxs]
    else:
        pairs = combinations(topics, 2)
    for list1, list2 in pairs:
        js = 1 - len(set(list1).intersection(set(list2)))/len(set(list1).union(set(list2)))
        dist = dist + js
        count = count + 1
    return dist/count


def pairwise_word_embedding_distance(topics, word_embedding_model, topk=10, sample=10000):
    """
    :param topk: how many most likely words to consider in the evaluation
    :return: topic coherence computed on the word embeddings similarities
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        count = 0
        sum_dist = 0
        num_topics = len(topics)
        num_combinations = (num_topics * (num_topics-1))/2
        if num_combinations > sample:
            pair_idxs = random_pairs(num_topics, sample)
            pairs = [(topics[i], topics[j]) for i, j in pair_idxs]
        else:
            pairs = combinations(topics, 2)
        for list1, list2 in pairs:
            count = count+1
            word_counts = 0
            dist = 0
            for word1 in list1[:topk]:
                for word2 in list2[:topk]:
                    dist = dist + distance.cosine(word_embedding_model[word1], word_embedding_model[word2])
                    word_counts = word_counts + 1

            dist = dist/word_counts
            sum_dist = sum_dist + dist
        return sum_dist/count


def centroid_distance(topics, word_embedding_model, topk=10, sample=10000):
    """
    :param topk: how many most likely words to consider in the evaluation
    :return: topic coherence computed on the word embeddings similarities
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        count = 0
        num_topics = len(topics)
        num_combinations = (num_topics * (num_topics-1))/2
        if num_combinations > sample:
            pair_idxs = random_pairs(num_topics, sample)
            pairs = [(topics[i], topics[j]) for i, j in pair_idxs]
        else:
            pairs = combinations(topics, 2)
        for list1, list2 in pairs:
            count = count + 1
            centroid1 = np.zeros(word_embedding_model.vector_size)
            centroid2 = np.zeros(word_embedding_model.vector_size)
            for word1 in list1[:topk]:
                centroid1 = centroid1 + word_embedding_model[word1]
            for word2 in list2[:topk]:
                centroid2 = centroid2 + word_embedding_model[word2]
            centroid1 = centroid1 / len(list1[:topk])
            centroid2 = centroid2 / len(list2[:topk])
        return distance.cosine(centroid1, centroid2)


def get_word2index(list1, list2):
    words = set(list1)
    words = words.union(set(list2))
    word2index = {w: i for i, w in enumerate(words)}
    return word2index
