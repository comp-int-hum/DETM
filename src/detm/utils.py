import copy
import logging
import random
import torch
import math
import numpy
from torch import autograd

logger = logging.getLogger("utils")

    
def train_model(
        subdocs,
        times,
        model,
        optimizer,
        max_epochs,
        clip=2.0,
        lr_factor=2.0,
        batch_size=32,
        device="cpu",
        val_proportion=0.2,
        detect_anomalies=False
):
    #times = [model.represent_time(t) for t in times]
    model = model.to(device)
    
    pairs = list(zip(subdocs, times))
    random.shuffle(pairs)
    
    train_subdocs = [x for x, _ in pairs[int(val_proportion*len(subdocs)):]]
    val_subdocs = [x for x, _ in pairs[:int(val_proportion*len(subdocs))]]

    train_times = [x for _, x in pairs[int(val_proportion*len(times)):]]
    val_times = [x for _, x in pairs[:int(val_proportion*len(times))]]
    
    best_state = copy.deepcopy(model.state_dict())
    best_val_ppl = float("inf")
    since_annealing = 0
    since_improvement = 0
    for epoch in range(1, max_epochs + 1):
        logger.info("Starting epoch %d", epoch)
        model.train(True)
        model.prepare_for_data(train_subdocs, train_times)
        
        acc_loss = 0
        acc_nll = 0
        acc_kl_theta_loss = 0
        acc_kl_eta_loss = 0
        acc_kl_alpha_loss = 0
        cnt = 0
        indices = torch.randperm(len(train_subdocs))
        indices = torch.split(indices, batch_size)
        for idx, ind in enumerate(indices):
            optimizer.zero_grad()
            model.zero_grad()
            actual_batch_size = len(ind)
            data_batch = numpy.zeros((actual_batch_size, model.vocab_size))
            times_batch = numpy.zeros((actual_batch_size, ))

            for i, doc_id in enumerate(ind):
                subdoc = train_subdocs[doc_id]
                times_batch[i] = train_times[doc_id] #0 if idx > 0 else train_times[doc_id]
                for k, v in subdoc.items():
                    data_batch[i, k] = v
            data_batch = torch.from_numpy(data_batch).float()
            times_batch = torch.from_numpy(times_batch)
            sums = data_batch.sum(1).unsqueeze(1)
            normalized_data_batch = data_batch / sums
            with autograd.set_detect_anomaly(detect_anomalies):

                loss, nll, kl_alpha, kl_eta, kl_theta = model(
                    data_batch,
                    times_batch,
                )
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            acc_loss += torch.sum(loss).item()
            acc_nll += torch.sum(nll).item()
            acc_kl_theta_loss += torch.sum(kl_theta).item()
            acc_kl_eta_loss += torch.sum(kl_eta).item()
            acc_kl_alpha_loss += torch.sum(kl_alpha).item()
            cnt += data_batch.shape[0]

        cur_loss = round(acc_loss / cnt, 2) 
        cur_nll = round(acc_nll / cnt, 2) 
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
        cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
        cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
        lr = optimizer.param_groups[0]['lr']


        logger.info("Computing perplexity...")
        _, val_ppl = apply_model(
            model,
            val_subdocs,
            val_times,
            batch_size,
            detect_anomalies=detect_anomalies
        )
        logger.info(
            '{}: LR: {}, Train doc losses: mix_prior={:.3f}, mix={:.3f}, embs={:.3f}, recon={:.3f}, NELBO={:.3f} Val doc ppl: {:.3f}'.format(
                epoch,
                lr,
                cur_kl_eta,
                cur_kl_theta,
                cur_kl_alpha,
                cur_nll,
                cur_loss,
                val_ppl
            )
        )

        if val_ppl < best_val_ppl:
            logger.info("Copying new best model...")
            best_val_ppl = val_ppl
            best_state = copy.deepcopy(model.state_dict())
            since_improvement = 0
            logger.info("Copied.")
        else:
            since_improvement += 1
        since_annealing += 1
        if since_improvement > 5 and since_annealing > 5 and since_improvement < 10:
            optimizer.param_groups[0]['lr'] /= lr_factor
            model.load_state_dict(best_state)
            since_annealing = 0
        elif numpy.isnan(val_ppl):
            logger.error("Perplexity was NaN: reducing learning rate and trying again...")
            model.load_state_dict(best_state)
            optimizer.param_groups[0]['lr'] /= lr_factor
        elif since_improvement >= 10:
            break

    return best_state

def test_for_lr(
        subdocs,
        times,
        model_class,
        word_list,
        num_topics, window_size,
        learning_rate, wdecay,
        embeddings,
        clip=2.0,
        lr_factor=2.0,
        batch_size=32,
        device="cpu",
        optimizer_type='adam',
        val_proportion=0.2,
        detect_anomalies=False
):
    
    # initialize data
    pairs = list(zip(subdocs, times))
    random.shuffle(pairs)
    
    train_subdocs = [x for x, _ in pairs[int(val_proportion*len(subdocs)):]]
    train_times = [x for _, x in pairs[int(val_proportion*len(times)):]]
    indices = torch.randperm(len(train_subdocs))
    indices = torch.split(indices, batch_size)

    # val_subdocs = [x for x, _ in pairs[:int(val_proportion*len(subdocs))]]
    # val_times = [x for _, x in pairs[:int(val_proportion*len(times))]]
    
    nan_flag = True
    while nan_flag:
        logger.info(f"starting new trial. Current learning rate {learning_rate}")

        nan_flag = False
        model = model_class(
        word_list=word_list,
        num_topics=num_topics,
        window_size=window_size,
        min_time=min(times),
        max_time=max(times),
        embeddings=embeddings,
    )

        model = model.to(device)
        model.prepare_for_data(train_subdocs, train_times)

        optimizer = (torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wdecay) 
                     if optimizer_type == 'adam' 
                     else torch.optim.SGD(model.parameters(), lr=learning_rate,  weight_decay=wdecay))
  
        for batch_idx, ind in enumerate(indices):
            if nan_flag:
                del model
                del optimizer
                break
            optimizer.zero_grad()
            model.zero_grad()
            actual_batch_size = len(ind)
            data_batch = numpy.zeros((actual_batch_size, model.vocab_size))
            times_batch = numpy.zeros((actual_batch_size, ))

            for i, doc_id in enumerate(ind):
                subdoc = train_subdocs[doc_id]
                times_batch[i] = train_times[doc_id] #0 if idx > 0 else train_times[doc_id]
                for k, v in subdoc.items():
                    data_batch[i, k] = v
            data_batch = torch.from_numpy(data_batch).float()
            times_batch = torch.from_numpy(times_batch)
            # sums = data_batch.sum(1).unsqueeze(1)
            # normalized_data_batch = data_batch / sums
            with autograd.set_detect_anomaly(detect_anomalies):

                loss, nll, kl_alpha, kl_eta, kl_theta = model(
                    data_batch,
                    times_batch,
                )
                logger.info(
                    'current batch {}: mix_prior={:.3f}, mix={:.3f}, embs={:.3f}, recon={:.3f}, NELBO={:.3f}'.format(
                        batch_idx,
                        torch.sum(kl_eta).item(),
                        torch.sum(kl_theta).item(),
                        torch.sum(kl_alpha).item(),
                        torch.sum(nll).item(),
                        torch.sum(loss).item()
                    )
                )                
                if torch.isnan(loss).any():
                    
                    logger.info(f"got nan in batch #{batch_idx} with lr {learning_rate}, reducing ...")
                    learning_rate /= lr_factor 
                    nan_flag = True
                    break
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            acc_loss += torch.sum(loss).item()
            acc_nll += torch.sum(nll).item()
            acc_kl_theta_loss += torch.sum(kl_theta).item()
            acc_kl_eta_loss += torch.sum(kl_eta).item()
            acc_kl_alpha_loss += torch.sum(kl_alpha).item()
            cnt += data_batch.shape[0]

        if not nan_flag:
            cur_loss = round(acc_loss / cnt, 2) 
            cur_nll = round(acc_nll / cnt, 2) 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
            cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
            lr = optimizer.param_groups[0]['lr']
            nan_flag = False
        
            logger.info(
                'LR: {}, Train doc losses: mix_prior={:.3f}, mix={:.3f}, embs={:.3f}, recon={:.3f}, NELBO={:.3f}'.format(
                    lr,
                    cur_kl_eta,
                    cur_kl_theta,
                    cur_kl_alpha,
                    cur_nll,
                    cur_loss
                )
            )

    return learning_rate

def apply_model(
        model,
        subdocs,
        times,
        batch_size=32,
        device="cpu",
        detect_anomalies=False
):
    model.train(False)
    model.prepare_for_data(subdocs, times)

    ppl = 0
    cnt = 0
    indices = torch.randperm(len(subdocs))
    indices = torch.split(indices, batch_size)

    for idx, ind in enumerate(indices):
        actual_batch_size = len(ind)
        data_batch = numpy.zeros((actual_batch_size, model.vocab_size))
        times_batch = numpy.zeros((actual_batch_size, ))

        for i, subdoc_id in enumerate(ind):
            subdoc = subdocs[subdoc_id]
            tm = times[subdoc_id]
            times_batch[i] = tm
            for k, v in subdoc.items():
                data_batch[i, k] = v
        data_batch = torch.from_numpy(data_batch).float()
        times_batch = torch.from_numpy(times_batch)
        sums = data_batch.sum(1).unsqueeze(1)
        with autograd.set_detect_anomaly(detect_anomalies):
            loss, nll, kl_alpha, kl_eta, kl_theta = model(
                data_batch,
                times_batch,
            )

            ppl += torch.sum(nll).item()
            cnt += data_batch.shape[0]
    return (), ppl / cnt

def get_matrice(model, subdocs, times, auxiliaries,
                batch_size, logger=None, detect_anomalies=False):

    if logger:
        logger.info(f"----- starts applying model ------ ")

    min_time, max_time, window_size = model.min_time, model.max_time, model.window_size
    num_topics, token_list = model.num_topics, model.word_list
    subdoc_topics = [[
        (token_list[idx], idx) for idx, _ in subdoc.items()] 
        for subdoc in subdocs]
        
    model.train(False)
    model.prepare_for_data(subdocs, times)

    indices = torch.randperm(len(subdocs))
    indices = torch.split(indices, batch_size)

    for inds in indices:
        actual_batch_size = len(inds)
        data_batch = numpy.zeros((actual_batch_size, model.vocab_size))
        times_batch = numpy.zeros((actual_batch_size, ))

        for i, subdoc_id in enumerate(inds):
            subdoc = subdocs[subdoc_id]
            tm = times[subdoc_id]
            times_batch[i] = tm
            for k, v in subdoc.items():
                data_batch[i, k] = v
        data_batch = torch.from_numpy(data_batch).float()
        times_batch = torch.from_numpy(times_batch)
        with autograd.set_detect_anomaly(detect_anomalies):
            liks = model.get_likelihood(data_batch, times_batch)
            
            try:
                for lik, ind in zip(liks, inds):
                    lik = lik.argmax(0)
                    subdoc_topics[ind] = [
                        (tok, lik[idx].item())
                        for tok, idx in subdoc_topics[ind]
                        ]
                
            except Exception as e:
                if logger:
                    logger.info(f"received error when enumerating text : {str(e)}")
                raise Exception(e)

    if logger:
        logger.info(f"----- completes applying model ------ ")
        logger.info(f"----- starts getting matrice ------ ")
        
    docs, doc2title, doc2author, doc2year = {}, {}, {}, {}
    unique_authors = set()
    token2id = {tok: idx for idx, tok in enumerate(token_list)}
        
    for subdoc, auxiliary in zip(subdoc_topics, auxiliaries):
        title = auxiliary["title"]
        author = auxiliary["author_info"]
        year = auxiliary["written_year"]
        htid = auxiliary["htid"]
        doc2title[htid] = title
        doc2author[htid] = author
        doc2year[htid] = year
        docs.setdefault(htid, [])
        docs[htid].append(subdoc)
        unique_authors.add(author)
    start_finish_per_window = [(min_time + idx * window_size, 
                            (min_time + (idx + 1) * window_size
                            if (min_time + (idx + 1) * window_size <= max_time) 
                            else max_time + 1)
                            )
                            for idx in range(math.ceil((max_time - min_time) / window_size))]
    window_rangs  = [numpy.arange(start_year, end_year)
                    for (start_year, end_year) in start_finish_per_window]
    # print(window_rangs)
    nwins, nwords, ntopics, nauths, ndocs = (len(window_rangs), len(token2id), 
                                            num_topics, len(unique_authors), len(docs))
       

    author2id = {a : i for i, a in enumerate(unique_authors)}
    htid2id = {d : i for i, d in enumerate(docs.keys())}
    time2window = {}
    for idx, years in enumerate(window_rangs):
        time2window.update({year: idx for year in years})
    
    if logger:
        logger.info(
            f"Found {nwins} windows, {nwords} unique words, {ntopics} unique topics, " +
            f"{ndocs} unique documents, and {nauths} unique authors"
        )

    words_wins_topics = numpy.zeros(shape=(nwords, nwins, ntopics))
    auths_wins_topics = numpy.zeros(shape=(nauths, nwins, ntopics))
    htid_wins_topics = numpy.zeros(shape=(ndocs, nwins, ntopics))
    
    for htid, subdocs in docs.items():
        title = doc2title[htid]
        author = doc2author[htid]
        year = doc2year[htid]
        aid = author2id[author]
        win = time2window[year]
        did = htid2id[htid]
        
        for subdoc in subdocs:
            for word, topic in subdoc:
                if topic != None:
                    wid = token2id[word]
                    words_wins_topics[wid, win, topic] += 1
                    auths_wins_topics[aid, win, topic] += 1
                    htid_wins_topics[did, win, topic] += 1

    if logger:
        logger.info(f"----- completes getting matrice ------ ")

    return {"start_time": min_time,
            "end_time": max_time,
            "window_size": window_size,
            "id2author": {i : a for a, i in author2id.items()},
            "id2word": {i : w for w, i in token2id.items()},
            "id2htid": {i : d for d, i in htid2id.items()},
            "doc2title": doc2title,
            "doc2author": doc2author,
            "doc2year": doc2year,
            "wwt": words_wins_topics,
            "awt": auths_wins_topics,
            "hwt": htid_wins_topics
        }

def get_top_topic_info(matrice, top_n=5):
        min_time, max_time, window_size = matrice['start_time'], matrice['end_time'], matrice['window_size']
        start_finish_per_window = [(min_time + idx * window_size, 
                                (min_time + (idx + 1) * window_size
                                 if (min_time + (idx + 1) * window_size <= max_time) 
                                 else max_time)
                                 )
                                for idx in range(math.ceil((max_time - min_time) / window_size))]
        range_str_per_window = [f"{start_year}-{end_year}" 
                                for (start_year, end_year) in start_finish_per_window]
        
        nwords, nwins, ntopics = matrice['wwt'].shape
        
        def _get_prop_and_dist_helper(data_window_topic_col, id2data_col, top_n, epsilon=1e-6):

            data_window_topic, id2data = matrice[data_window_topic_col], matrice[id2data_col]
            return_data = {}

            data_dist_per_top = (data_window_topic.transpose(2, 0, 1) / 
                                   (data_window_topic.sum(2) + epsilon)).transpose(1, 2, 0)

            topic_dist_per_data = data_window_topic / (data_window_topic.sum(0) + epsilon)

            const_str = "Data proportion in single topic "

            for topic_idx in range(ntopics):
                for temp_idx in range(nwins):
                    score_per_topic_per_time = data_dist_per_top[:, temp_idx, topic_idx]
                    top_n_indices = numpy.argsort(score_per_topic_per_time)[-top_n:][::-1]
                    top_n_data = score_per_topic_per_time[top_n_indices]
                    key = const_str + f"#{topic_idx} time {range_str_per_window[temp_idx]}"
                    return_data[key] = {
                        f"# {idx}": {'score': round(top_n_data[idx], 4), 
                                    'name': id2data[top_n_indices[idx]]}
                        for idx in range(top_n)}
        
            # only per topic
            for topic_idx in range(ntopics):
                # summing over the time axis
                score_per_topic = (data_dist_per_top.sum(1))[:, topic_idx]
                top_n_indices = numpy.argsort(score_per_topic)[-top_n:][::-1]
                top_n_data = score_per_topic[top_n_indices]
                key = const_str + f"#{topic_idx} across all window"
                return_data[key] = {
                        f"# {idx}": {'score': round(top_n_data[idx], 4), 
                                    'name': id2data[top_n_indices[idx]]}
                        for idx in range(top_n)}
        
            const_str = "Topic proportion in single data "

            for topic_idx in range(ntopics):
                for temp_idx in range(nwins):
                    score_per_topic_per_time = topic_dist_per_data[:, temp_idx, topic_idx]
                    top_n_indices = numpy.argsort(score_per_topic_per_time)[-top_n:][::-1]
                    top_n_data = score_per_topic_per_time[top_n_indices]
                    key = const_str + f"#{topic_idx} time {range_str_per_window[temp_idx]}"
                    return_data[key] = {
                        f"# {idx}": {'score': round(top_n_data[idx], 4), 
                                    'name': id2data[top_n_indices[idx]]}
                        for idx in range(top_n)}
        
            # only per topic
            for topic_idx in range(ntopics):
                # summing over the time axis
                score_per_topic = (topic_dist_per_data.sum(1))[:, topic_idx]
                top_n_indices = numpy.argsort(score_per_topic)[-top_n:][::-1]
                top_n_data = score_per_topic[top_n_indices]
                key = const_str + f"#{topic_idx} across all window"
                return_data[key] = {
                        f"# {idx}": {'score': round(top_n_data[idx], 4), 
                                    'name': id2data[top_n_indices[idx]]}
                        for idx in range(top_n)}
        
            return return_data
        
        data = {}
        data['per work'] = _get_prop_and_dist_helper("hwt", "id2htid", top_n)
        data['per vocab'] = _get_prop_and_dist_helper("wwt", "id2word", top_n)
        data['per author'] = _get_prop_and_dist_helper("awt", "id2author", top_n)
        return data

# from sklearn.manifold import TSNE
# import torch 
# import numpy as numpy
# import bokeh.plotting as bp
# from bokeh.plotting import save
# from bokeh.models import HoverTool
# import matplotlib.pyplot as plt 
# import matplotlib 

# tiny = 1e-6

# def _reparameterize(mu, logvar, num_samples):
#     """Applies the reparameterization trick to return samples from a given q"""
#     std = torch.exp(0.5 * logvar) 
#     bsz, zdim = logvar.size()
#     eps = torch.randn(num_samples, bsz, zdim).to(mu.device)
#     mu = mu.unsqueeze(0)
#     std = std.unsqueeze(0)
#     res = eps.mul_(std).add_(mu)
#     return res

# def get_document_frequency(data, wi, wj=None):
#     if wj is None:
#         D_wi = 0
#         for l in range(len(data)):
#             doc = data[l].squeeze(0)
#             if len(doc) == 1: 
#                 continue
#                 #doc = [doc.squeeze()]
#             else:
#                 doc = doc.squeeze()
#             if wi in doc:
#                 D_wi += 1
#         return D_wi
#     D_wj = 0
#     D_wi_wj = 0
#     for l in range(len(data)):
#         doc = data[l].squeeze(0)
#         if len(doc) == 1: 
#             doc = [doc.squeeze()]
#         else:
#             doc = doc.squeeze()
#         if wj in doc:
#             D_wj += 1
#             if wi in doc:
#                 D_wi_wj += 1
#     return D_wj, D_wi_wj 

# def get_topic_coherence(beta, data, vocab):
#     D = len(data) ## number of docs...data is list of documents
#     print('D: ', D)
#     TC = []
#     num_topics = len(beta)
#     for k in range(num_topics):
#         print('k: {}/{}'.format(k, num_topics))
#         top_10 = list(beta[k].argsort()[-11:][::-1])
#         top_words = [vocab[a] for a in top_10]
#         TC_k = 0
#         counter = 0
#         for i, word in enumerate(top_10):
#             # get D(w_i)
#             D_wi = get_document_frequency(data, word)
#             j = i + 1
#             tmp = 0
#             while j < len(top_10) and j > i:
#                 # get D(w_j) and D(w_i, w_j)
#                 D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
#                 # get f(w_i, w_j)
#                 if D_wi_wj == 0:
#                     f_wi_wj = -1
#                 else:
#                     f_wi_wj = -1 + ( numpy.log(D_wi) + numpy.log(D_wj)  - 2.0 * numpy.log(D) ) / ( numpy.log(D_wi_wj) - numpy.log(D) )
#                 # update tmp: 
#                 tmp += f_wi_wj
#                 j += 1
#                 counter += 1
#             # update TC_k
#             TC_k += tmp 
#         TC.append(TC_k)
#     print('counter: ', counter)
#     print('num topics: ', len(TC))
#     #TC = numpy.mean(TC) / counter
#     print('Topic Coherence is: {}'.format(TC))
#     return TC, counter

# def log_gaussian(z, mu=None, logvar=None):
#     sz = z.size()
#     d = z.size(2)
#     bsz = z.size(1)
#     if mu is None or logvar is None:
#         mu = torch.zeros(bsz, d).to(z.device)
#         logvar = torch.zeros(bsz, d).to(z.device)
#     mu = mu.unsqueeze(0)
#     logvar = logvar.unsqueeze(0)
#     var = logvar.exp()
#     log_density = ((z - mu)**2 / (var+tiny)).sum(2) # b
#     log_det = logvar.sum(2) # b
#     log_density = log_density + log_det + d*numpy.log(2*numpy.pi)
#     return -0.5*log_density

# def logsumexp(x, dim=0):
#     d = torch.max(x, dim)[0]   
#     if x.dim() == 1:
#         return torch.log(torch.exp(x - d).sum(dim)) + d
#     else:
#         return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim) + tiny) + d

# def flatten_docs(docs): #to get words and doc_indices
#     words = [x for y in docs for x in y]
#     doc_indices = [[j for _ in doc] for j, doc in enumerate(docs)]
#     doc_indices = [x for y in doc_indices for x in y]
#     return words, doc_indices
    
# def onehot(data, min_length):
#     return list(numpy.bincount(data, minlength=min_length))

# def nearest_neighbors(word, embeddings, vocab, num_words):
#     vectors = embeddings.cpu().numpy() 
#     index = vocab.index(word)
#     query = embeddings[index].cpu().numpy() 
#     ranks = vectors.dot(query).squeeze()
#     denom = query.T.dot(query).squeeze()
#     denom = denom * numpy.sum(vectors**2, 1)
#     denom = numpy.sqrt(denom)
#     ranks = ranks / denom
#     mostSimilar = []
#     [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
#     nearest_neighbors = mostSimilar[:num_words]
#     nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
#     return nearest_neighbors

# def visualize(docs, _lda_keys, topics, theta):
#     tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
#     # project to 2D
#     tsne_lda = tsne_model.fit_transform(theta)
#     colormap = []
#     for name, hex in matplotlib.colors.cnames.items():
#         colormap.append(hex)

#     colormap = colormap[:len(theta[0, :])]
#     colormap = numpy.array(colormap)

#     title = '20 newsgroups TE embedding V viz'
#     num_example = len(docs)

#     plot_lda = bp.figure(plot_width=1400, plot_height=1100,
#                      title=title,
#                      tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
#                      x_axis_type=None, y_axis_type=None, min_border=1)

#     plt.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1],
#                  color=colormap[_lda_keys][:num_example])
#     plt.show()
