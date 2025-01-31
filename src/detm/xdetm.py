"""This file defines a dynamic etm object.
"""
import math
import logging
import torch
import numpy
from .abstract_detm import AbstractDETM


logger = logging.getLogger("xdetm")


class xDETM(AbstractDETM):
    def __init__(
            self,
            num_topics,
            embeddings,
            window_size,
            min_time,
            max_time,
            word_list,
            t_hidden_size=800,
            eta_hidden_size=200,
            enc_drop=0.0,
            eta_dropout=0.0,
            eta_nlayers=3,
            delta=0.005,
    ):
        super(xDETM, self).__init__(num_topics, word_list, embeddings)        
        self.max_time = max_time
        self.min_time = min_time
        self.window_size = window_size
        self.num_windows = math.ceil((max_time - min_time) / window_size)
        self.t_hidden_size = t_hidden_size
        self.eta_hidden_size = eta_hidden_size
        self.enc_drop = enc_drop
        self.eta_nlayers = eta_nlayers
        self.t_drop = torch.nn.Dropout(enc_drop)
        self.delta = delta

        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
        self.mu_q_alpha = torch.nn.Parameter(torch.randn(self.num_topics, self.num_windows, self.embedding_size))
        self.logsigma_q_alpha = torch.nn.Parameter(torch.randn(self.num_topics, self.num_windows, self.embedding_size))
    
        ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D
        self.q_theta = torch.nn.Sequential(
                    torch.nn.Linear(self.vocab_size+self.num_topics, t_hidden_size), 
                    torch.nn.RReLU(), #self.theta_act,
                    torch.nn.Linear(t_hidden_size, t_hidden_size),
                    torch.nn.RReLU()
                )
        self.mu_q_theta = torch.nn.Linear(t_hidden_size, self.num_topics, bias=True)
        self.logsigma_q_theta = torch.nn.Linear(t_hidden_size, self.num_topics, bias=True)

        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = torch.nn.Linear(self.vocab_size, eta_hidden_size)
        self.q_eta = torch.nn.LSTM(eta_hidden_size, eta_hidden_size, eta_nlayers, dropout=eta_dropout)
        self.mu_q_eta = torch.nn.Linear(eta_hidden_size+self.num_topics, self.num_topics, bias=True)
        self.logsigma_q_eta = torch.nn.Linear(eta_hidden_size+self.num_topics, self.num_topics, bias=True)
        logger.info("%d windows, %d topics, %d words", self.num_windows, self.num_topics, self.vocab_size)
        
    def represent_time(self, time):
        return int((time - self.min_time) / self.window_size)

    def topic_embeddings(self, document_times):
        alphas = torch.zeros(self.num_windows, self.num_topics, self.embedding_size).to(self.device)
        kl_alpha = []
        alphas[0] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])
        p_mu_0 = torch.zeros(self.num_topics, self.embedding_size).to(self.device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.embedding_size).to(self.device)
        kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)
        kl_alpha.append(kl_0)
        for t in range(1, self.num_windows):
            alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :]) 
            p_mu_t = alphas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.embedding_size).to(self.device))
            kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)
            kl_alpha.append(kl_t)
        return alphas[document_times], torch.stack(kl_alpha)

    def document_topic_mixture_priors(self, document_times):
        inp = self.q_eta_map(self.rnn_input).unsqueeze(1)
        #hidden = self.init_hidden()
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        hidden = (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))

        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze()
        etas = torch.zeros(self.num_windows, self.num_topics).to(self.device)
        kl_eta = []
        inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(self.device)], dim=0)
        mu_0 = self.mu_q_eta(inp_0)
        if self.training:
            logsigma_0 = self.logsigma_q_eta(inp_0)
            etas[0] = self.reparameterize(mu_0, logsigma_0)
            p_mu_0 = torch.zeros(self.num_topics,).to(self.device)
            logsigma_p_0 = torch.zeros(self.num_topics,).to(self.device)
            kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
            kl_eta.append(kl_0)
        else:
            etas[0] = mu_0
            kl_eta.append(torch.tensor([]).to(self.device))
        for t in range(1, self.num_windows):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            if self.training:
                logsigma_t = self.logsigma_q_eta(inp_t)
                etas[t] = self.reparameterize(mu_t, logsigma_t)
                p_mu_t = etas[t-1]
                logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics,).to(self.device))
                kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
                kl_eta.append(kl_t)
            else:
                etas[t] = mu_t
                kl_eta.append(torch.tensor([]).to(self.device))
        return etas[document_times], torch.stack(kl_eta)
    
    def document_topic_mixtures(self, document_topic_mixture_priors, document_word_counts, document_times):
        """Returns the topic proportions.
        """
        inp = torch.cat([document_word_counts, document_topic_mixture_priors], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0 and self.training:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        if not self.training:
            return torch.nn.functional.softmax(mu_theta, dim=-1), torch.tensor([]).to(self.device)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = torch.nn.functional.softmax(z, dim=-1)                
        kl_theta = self.get_kl(mu_theta, logsigma_theta, document_topic_mixture_priors, torch.zeros(self.num_topics).to(self.device))
        return theta, kl_theta

    def prepare_for_data(self, document_word_counts, document_times, batch_size=1024):
        self.num_docs = len(document_word_counts)
        document_times = [self.represent_time(t) for t in document_times]
        window_count = self.num_windows
        indices = torch.arange(0, len(document_word_counts), dtype=torch.int)
        indices = torch.split(indices, batch_size)
        rnn_input = torch.zeros(window_count, self.vocab_size).to(self.device)
        cnt = torch.zeros(window_count, ).to(self.device)
        for idx, ind in enumerate(indices):
            batch_size = len(ind)
            data_batch = numpy.zeros((batch_size, self.vocab_size))
            times_batch = numpy.zeros((batch_size, ))
            for i, doc_id in enumerate(ind):
                subdoc = document_word_counts[doc_id]
                window = document_times[doc_id]
                times_batch[i] = window
                for k, v in subdoc.items():
                    data_batch[i, k] = v
            data_batch = torch.from_numpy(data_batch).float().to(self.device)
            times_batch = torch.from_numpy(times_batch).to(self.device)
            for t in range(window_count):
                tmp = (times_batch == t).nonzero()
                docs = data_batch[tmp].squeeze().sum(0)
                rnn_input[t] += docs
                cnt[t] += len(tmp)
        self.rnn_input = rnn_input / cnt.unsqueeze(1)
        nan_indices = torch.isnan(self.rnn_input).nonzero(as_tuple=True)
        logger.info(f"NaN values found at window: {set(nan_indices[0].tolist())}")