import copy
import logging
import random
import torch
import numpy
from torch import autograd
from collections import Counter
from tqdm import tqdm
from typing import List, Dict

# logging.basicConfig(filename='app.log', level=logging.INFO)
logger = logging.getLogger("utils")

def _yield_data(subdocs, times, vocab_size, batch_size=64):
    word_count = 0
    indices = torch.randperm(len(subdocs))
    indices = torch.split(indices, batch_size)

    for ind in tqdm(indices):
        actual_batch_size = len(ind)
        data_batch = numpy.zeros((actual_batch_size, vocab_size))
        times_batch = numpy.zeros((actual_batch_size, ))

        for i, doc_id in enumerate(ind):
            subdoc = subdocs[doc_id]
            times_batch[i] = times[doc_id]
            for k, v in subdoc.items():
                data_batch[i, k] = v
                word_count += v
        data_batch = torch.from_numpy(data_batch).float()
        times_batch = torch.from_numpy(times_batch)

        yield times_batch, data_batch, word_count, ind
    
def train_model(
        subdocs, times,
        model, optimizer,
        max_epochs, clip=10.0,
        lr_factor=2.0, batch_size=32,
        device="cpu", val_proportion=0.2,
        detect_anomalies=False, use_wandb=False):

    model = model.to(device)
    
    pairs = list(zip(subdocs, times))
    random.shuffle(pairs)
    
    train_subdocs = [x for x, _ in pairs[int(val_proportion*len(subdocs)):]]
    val_subdocs = [x for x, _ in pairs[:int(val_proportion*len(subdocs))]]

    train_times = [x for _, x in pairs[int(val_proportion*len(times)):]]
    val_times = [x for _, x in pairs[:int(val_proportion*len(times))]]

    train_time_wins = [int((time_instance - model.min_time) / model.window_size) for time_instance in train_times]
    counter_info = dict(sorted(Counter(train_time_wins).items()))
    logger.info(f"information of instances present per set: {counter_info}")
    
    logger.info("Saving initial model parameters")
    best_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_val_ppl = float("inf")
    since_annealing = 0
    since_improvement = 0
    
    
    for epoch in range(1, max_epochs + 1):
        logger.info("Starting epoch %d", epoch)
        model.train(True)
        logger.info("Preparing for data")
        model.prepare_for_data(train_subdocs, train_times)
        
        acc_loss = 0
        acc_nll = 0
        acc_kl_theta_loss = 0
        acc_kl_eta_loss = 0
        acc_kl_alpha_loss = 0
        cnt = 0
        word_count = 0

        logger.info("Computing batches")
        train_generator = _yield_data(train_subdocs, train_times, model.vocab_size,
                                      batch_size)

        logger.info("Processing training and updating model")
        while True:
            try:
                times_batch, data_batch, word_subcount, _ = next(train_generator)
                word_count += word_subcount
                optimizer.zero_grad()
                model.zero_grad()

                with autograd.set_detect_anomaly(detect_anomalies):

                    loss, nll, kl_alpha, kl_eta, kl_theta = model(data_batch, times_batch)
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
            
            except StopIteration:
                break

        cur_loss = acc_loss / word_count
        cur_nll = acc_nll / word_count
        cur_kl_theta = acc_kl_theta_loss / word_count
        cur_kl_eta = acc_kl_eta_loss / word_count 
        cur_kl_alpha = acc_kl_alpha_loss / word_count
        lr = optimizer.param_groups[0]['lr']


        logger.info("Processing validation")
        _, val_ppl = apply_model(
            model,
            val_subdocs,
            val_times,
            batch_size,
            detect_anomalies=detect_anomalies
        )
        logger.info(
            '{}: LR: {}, Train loss per word: mix_prior={:.2f}, mix={:.2f}, embs={:.2f}, recon={:.2f}, NELBO={:.2f} Val ppl per word: {:.2f}'.format(
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
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
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
            optimizer.load_state_dict(best_optimizer_state)
            model.load_state_dict(best_state)
            optimizer.param_groups[0]['lr'] /= lr_factor
        elif since_improvement >= 10:
            break

    return best_state


def apply_model(
        model, subdocs,
        times, batch_size=32, device="cpu", 
        detect_anomalies=False, use_wandb=False
):
    model.train(False)
    logger.info("Preparing for data")
    model.prepare_for_data(subdocs, times)

    ppl = 0
    cnt = 0
    word_count = 0
    
    appl_generator = _yield_data(subdocs, times, model.vocab_size, batch_size)
    while True:
        try:
            times_batch, data_batch, word_subcount, _ = next(appl_generator)
            word_count += word_subcount
            with autograd.set_detect_anomaly(detect_anomalies):
                _, nll, _, _, _ = model(
                    data_batch,
                    times_batch,
                )

                ppl += torch.sum(nll).item()
                cnt += data_batch.shape[0]
        except StopIteration:
            break

            ppl += torch.sum(nll).item()
            cnt += data_batch.shape[0]

    return (), ppl / word_count

class AuthorData:

    def __init__(self, name : str, num_topics: int,
                works: List[str] = [], vocabs: List[str] = []):
        self.name = name
        self.work2idx = {work : idx for idx, work in enumerate(works)}
        # here ideally vocabs is a list of word that the author has used based on an overall vocab_list
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(vocabs)}
        self.annotate_by_vocab = numpy.zeros((len(vocabs), len(works), num_topics))
        self.annotate_by_work =  numpy.zeros((len(works), num_topics))
        
    def annotate_subdocs(self, work, document_topic_mixture, per_vocab_topic_class):
        work_idx = self.work2idx[work]
        self.annotate_by_work[work_idx] += document_topic_mixture

        for vocab, topic_class in per_vocab_topic_class.items():
            self.annotate_by_vocab[self.vocab2idx[vocab], work_idx, topic_class] += 1

def annotate_data(model, subdocs, times, auxiliaries,
                author_field, title_field,
                vocab_field="vocab_used",
                batch_size=32, device="cpu", 
                detect_anomalies=False, use_wandb=False):

    num_topics, vocab_list = model.num_topics, model.word_list

    auth2work_vocab_tuple = {}
    for auxiliary, time in zip(auxiliaries, times):
        author_name = auxiliary[author_field]
        title_name_w_year = auxiliary[title_field] + f" [-] " + str(int(time))
        title_set, vocab_set = auth2work_vocab_tuple.get(author_name, (set(), set()))
        title_set.add(title_name_w_year)
        vocab_set.update(auxiliary[vocab_field])
        auth2work_vocab_tuple[author_name] = (title_set, vocab_set)
    
    auth2annote = {}
    for auth_name, (title_set, vocab_set) in auth2work_vocab_tuple.items():
        auth2annote[auth_name] = AuthorData(auth_name, num_topics, works=list(title_set), vocabs=list(vocab_set))
    
    del auth2work_vocab_tuple

    model.train(False)
    logger.info("Preparing for data")
    model.prepare_for_data(subdocs, times)
    
    appl_generator = _yield_data(subdocs, times, model.vocab_size, batch_size)

    while True:
        try:         
            times_batch, data_batch, _, inds = next(appl_generator)
            with autograd.set_detect_anomaly(detect_anomalies):
                document_topic_mixtures, vocab_topic_mixtures = model(
                    data_batch, times_batch, annotate=True)
                document_topic_mixtures = document_topic_mixtures.cpu().detach().numpy()
                vocab_topic_mixtures = vocab_topic_mixtures.cpu().detach().numpy()
                
                for ind, data_ins, document_topic_mixture, vocab_topic_mixture in zip(inds, data_batch, 
                                                            document_topic_mixtures, vocab_topic_mixtures):
                    vocab_topic = vocab_topic_mixture.argmax(0)
                    token_instances = (torch.nonzero(data_ins, as_tuple=True)[0]).cpu().detach().numpy()
                    per_vocab_topic_class = {vocab_list[tok_id] : vocab_topic[tok_id] for tok_id in token_instances}
                    auxiliary, time = auxiliaries[ind], times[ind]
                    author_name, title = auxiliary[author_field], auxiliary[title_field] + " [-] " + str(int(time))
                    auth2annote[author_name].annotate_subdocs(title, document_topic_mixture, per_vocab_topic_class)
                    
        except StopIteration:
            break
    del appl_generator, model, subdocs, times, auxiliaries
    logger.info(f"completing annotation, organizing data to be returned: ")
    auth_matrix = [{
        "auth_name": auth_name,
        "work2idx": auth_data.work2idx,
        "vocab2idx": auth_data.vocab2idx,
        "annotate_by_vocab": auth_data.annotate_by_vocab.tolist(),
        "annotate_by_work": auth_data.annotate_by_work.tolist()
    } for auth_name, auth_data in auth2annote.items()]
    del auth2annote
    logger.info("completes creating matrix")
    return auth_matrix