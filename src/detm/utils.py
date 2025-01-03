import copy
import logging
import random
import torch
import numpy


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
        val_proportion=0.2
):
    #word_to_id = {w : i for i, w in enumerate(model.embeddings.wv.index_to_key)}

    #subdocs = [{model.word_to_id[w] : c for w, c in subdoc.items()} for subdoc in subdocs]
    times = [model.represent_time(t) for t in times]
    
    random.shuffle(subdocs)
    train_docs = subdocs[int(val_proportion*len(subdocs)):]
    val_docs = subdocs[:int(val_proportion*len(subdocs))]

    train_times = times[int(val_proportion*len(times)):]
    val_times = times[:int(val_proportion*len(times))]
    
    train_rnn_input = model.get_rnn_input(train_docs, train_times)
    val_rnn_input = model.get_rnn_input(train_docs, train_times)

    best_state = None
    best_val_ppl = None
    since_annealing = 0
    since_improvement = 0
    for epoch in range(1, max_epochs + 1):
        logger.info("Starting epoch %d", epoch)
        model.train()
        
        acc_loss = 0
        acc_nll = 0
        acc_kl_theta_loss = 0
        acc_kl_eta_loss = 0
        acc_kl_alpha_loss = 0
        cnt = 0
        indices = torch.randperm(len(train_docs))
        indices = torch.split(indices, batch_size)

        for idx, ind in enumerate(indices):
            optimizer.zero_grad()
            model.zero_grad()
            batch_size = len(ind)
            data_batch = numpy.zeros((batch_size, model.vocab_size))
            times_batch = numpy.zeros((batch_size, ))

            for i, doc_id in enumerate(ind):
                subdoc = train_docs[doc_id]
                tm = train_times[doc_id]
                times_batch[i] = tm
                #data_batch.append(subdoc)
                for k, v in subdoc.items():
                    data_batch[i, k] = v
            data_batch = torch.from_numpy(data_batch).float()
            times_batch = torch.from_numpy(times_batch)
            sums = data_batch.sum(1).unsqueeze(1)
            normalized_data_batch = data_batch / sums
            loss, nll, kl_alpha, kl_eta, kl_theta = model(
                data_batch,
                normalized_data_batch,
                times_batch,
                train_rnn_input,
                len(train_docs),
            )
            if not torch.any(torch.isnan(loss)):
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                acc_loss += torch.sum(loss).item()
                acc_nll += torch.sum(nll).item()
                acc_kl_theta_loss += torch.sum(kl_theta).item()
                acc_kl_eta_loss += torch.sum(kl_eta).item()
                acc_kl_alpha_loss += torch.sum(kl_alpha).item()
            cnt += 1

        cur_loss = round(acc_loss / cnt, 2) 
        cur_nll = round(acc_nll / cnt, 2) 
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
        cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
        cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
        lr = optimizer.param_groups[0]['lr']


        logger.info("Computing perplexity...")

        val_ppl = model.get_completion_ppl(val_docs, val_times, val_rnn_input, device)
        logger.info(
            '{}: LR: {}, KL_theta: {}, KL_eta: {}, KL_alpha: {}, Rec_loss: {}, NELBO: {}, PPL: {}'.format(
                epoch,
                lr,
                cur_kl_theta,
                cur_kl_eta,
                cur_kl_alpha,
                cur_nll,
                cur_loss,
                val_ppl
            )
        )

        if best_val_ppl == None or val_ppl < best_val_ppl:
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
        elif since_improvement >= 10:
            break

    return copy.deepcopy(model.state_dict())


def perplexity_on_corpus(
        model,
        corpus,
        max_subdoc_length,
        content_field,
        time_field=None,
        lowercase=True,
        device="cpu"
):
    subdocs, times = corpus.filter_for_model(
        model,
        max_subdoc_length,
        content_field,
        time_field,
        lowercase
    )
    times = [model.represent_time(t) for t in times]
    rnn_input = model.get_rnn_input(subdocs, times)
    return model.get_completion_ppl(subdocs, times, rnn_input, device)


# from sklearn.manifold import TSNE
# import torch 
# import numpy as np
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
#                     f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
#                 # update tmp: 
#                 tmp += f_wi_wj
#                 j += 1
#                 counter += 1
#             # update TC_k
#             TC_k += tmp 
#         TC.append(TC_k)
#     print('counter: ', counter)
#     print('num topics: ', len(TC))
#     #TC = np.mean(TC) / counter
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
#     log_density = log_density + log_det + d*np.log(2*np.pi)
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
#     return list(np.bincount(data, minlength=min_length))

# def nearest_neighbors(word, embeddings, vocab, num_words):
#     vectors = embeddings.cpu().numpy() 
#     index = vocab.index(word)
#     query = embeddings[index].cpu().numpy() 
#     ranks = vectors.dot(query).squeeze()
#     denom = query.T.dot(query).squeeze()
#     denom = denom * np.sum(vectors**2, 1)
#     denom = np.sqrt(denom)
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
#     colormap = np.array(colormap)

#     title = '20 newsgroups TE embedding V viz'
#     num_example = len(docs)

#     plot_lda = bp.figure(plot_width=1400, plot_height=1100,
#                      title=title,
#                      tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
#                      x_axis_type=None, y_axis_type=None, min_border=1)

#     plt.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1],
#                  color=colormap[_lda_keys][:num_example])
#     plt.show()
