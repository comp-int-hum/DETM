"""This file defines a dynamic etm object.
"""
import math
import torch
import torch.nn.functional as F 
from torch import nn
import numpy


class DETM(nn.Module):
    def __init__(
            self,
            num_topics,
            min_time,
            max_time,
            word_list,
            embeddings,
            t_hidden_size=800,
            eta_hidden_size=200,
            enc_drop=0.0,
            eta_dropout=0.0,
            eta_nlayers=3,
            delta=0.005,
            window_size=None,
            train_embeddings=False,
            theta_act="relu",
            batch_size=32,
            device="cpu"
    ):
        super(DETM, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.word_list = word_list
        
        ## define hyperparameters
        self.num_topics = num_topics
        self.max_time = max_time
        self.min_time = min_time
        self.window_size = window_size
        self.num_windows = math.ceil((max_time - min_time) / window_size)
        self.t_hidden_size = t_hidden_size
        self.eta_hidden_size = eta_hidden_size

        self.enc_drop = enc_drop
        self.eta_nlayers = eta_nlayers
        self.t_drop = nn.Dropout(enc_drop)
        self.delta = delta
        self.train_embeddings = train_embeddings
        self.theta_act = self.get_activation(theta_act)

        
        rho_data = numpy.array([embeddings.wv[w] for w in self.word_list])
        num_embeddings, emsize = rho_data.shape
        self.emsize = emsize
        self.rho_size = self.emsize
        
        rho = nn.Embedding(num_embeddings, self.emsize)
        rho.weight.data = torch.tensor(rho_data)
        self.rho = rho.weight.data.clone().float().to(self.device)
        
        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
        self.mu_q_alpha = nn.Parameter(torch.randn(num_topics, self.num_windows, self.rho_size))
        self.logsigma_q_alpha = nn.Parameter(torch.randn(num_topics, self.num_windows, self.rho_size))
    
        ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D
        self.q_theta = nn.Sequential(
                    nn.Linear(self.vocab_size+num_topics, t_hidden_size), 
                    self.theta_act,
                    nn.Linear(t_hidden_size, t_hidden_size),
                    self.theta_act,
                )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = nn.Linear(self.vocab_size, eta_hidden_size)
        self.q_eta = nn.LSTM(eta_hidden_size, eta_hidden_size, eta_nlayers, dropout=eta_dropout)
        self.mu_q_eta = nn.Linear(eta_hidden_size+num_topics, num_topics, bias=True)
        self.logsigma_q_eta = nn.Linear(eta_hidden_size+num_topics, num_topics, bias=True)

    def represent_time(self, time):
        return int((time - self.min_time) / self.window_size)
        
    @property
    def vocab_size(self):
        return self.rho.shape[0]
        
    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl

    def get_alpha(self): ## mean field
        alphas = torch.zeros(self.num_windows, self.num_topics, self.rho_size).to(self.device)
        kl_alpha = []

        alphas[0] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])

        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)
        kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)
        kl_alpha.append(kl_0)
        for t in range(1, self.num_windows):
            alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :]) 
            
            p_mu_t = alphas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(self.device))
            kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)
            kl_alpha.append(kl_t)
        kl_alpha = torch.stack(kl_alpha).sum()
        return alphas, kl_alpha.sum()

    def get_eta(self, rnn_inp): ## structured amortized inference
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
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
        if self.training:
            kl_eta = torch.stack(kl_eta).sum()
            return etas, kl_eta
        else:
            return etas, None
    
    def get_theta(self, eta, bows, times): ## amortized inference
        """Returns the topic proportions.
        """
        eta_td = eta[times.type('torch.LongTensor')]
        inp = torch.cat([bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0 and self.training:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        if not self.training:
            return torch.nn.functional.softmax(mu_theta, dim=-1), None
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_td, torch.zeros(self.num_topics).to(self.device))
        return theta, kl_theta
    
    def get_beta(self, alpha):
        """Returns the topic matrix \beta of shape K x V
        """
        tmp = alpha.view(alpha.size(0)*alpha.size(1), self.rho_size)
        logit = torch.mm(tmp, self.rho.permute(1, 0)) 
        logit = logit.view(alpha.size(0), alpha.size(1), -1)
        beta = F.softmax(logit, dim=-1)
        return beta 

    def get_nll(self, theta, beta, bows):
        theta = theta.unsqueeze(1)
        loglik = torch.bmm(theta, beta).squeeze(1)
        loglik = loglik
        loglik = torch.log(loglik+1e-6)
        nll = -loglik * bows
        nll = nll.sum(-1)
        return nll
    
    def forward(self, bows, normalized_bows, times, rnn_inp, num_docs):
        
        bows = bows.to(self.device)
        normalized_bows = normalized_bows.to(self.device)
        times = times.to(self.device)
        rnn_inp = rnn_inp.to(self.device)

        bsz = normalized_bows.size(0)
        coeff = num_docs / bsz 
        alpha, kl_alpha = self.get_alpha()
        eta, kl_eta = self.get_eta(rnn_inp)
        theta, kl_theta = self.get_theta(eta, normalized_bows, times)
        kl_theta = kl_theta.sum() * coeff

        beta = self.get_beta(alpha)
        beta = beta[times.type('torch.LongTensor')]
        nll = self.get_nll(theta, beta, bows)
        nll = nll.sum() * coeff
        nelbo = nll + kl_alpha + kl_eta + kl_theta
        return nelbo, nll, kl_alpha, kl_eta, kl_theta

    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for \eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))

    def get_rnn_input(self, subdocs, times, batch_size=32):        
        window_count = self.num_windows
        indices = torch.arange(0, len(subdocs), dtype=torch.int)
        indices = torch.split(indices, batch_size)
        rnn_input = torch.zeros(window_count, self.vocab_size).to(self.device)
        cnt = torch.zeros(window_count, ).to(self.device)
        for idx, ind in enumerate(indices):
            batch_size = len(ind)
            data_batch = numpy.zeros((batch_size, self.vocab_size))
            times_batch = numpy.zeros((batch_size, ))
            for i, doc_id in enumerate(ind):
                subdoc = subdocs[doc_id]
                window = times[doc_id]
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
        rnn_input = rnn_input / cnt.unsqueeze(1)
        return rnn_input

    def get_completion_ppl(self, val_subdocs, val_times, val_rnn_inp, device, batch_size=128):
        """Returns document completion perplexity.
        """

        self.eval()
        with torch.no_grad():
            alpha = self.mu_q_alpha
            acc_loss = 0.0
            cnt = 0
            eta, _ = self.get_eta(val_rnn_inp)
            indices = torch.split(torch.tensor(range(len(val_subdocs))), batch_size)
            for idx, ind in enumerate(indices):
                batch_size = len(ind)
                data_batch = numpy.zeros((batch_size, self.vocab_size))
                times_batch = numpy.zeros((batch_size, ))
                for i, doc_id in enumerate(ind):
                    subdoc = val_subdocs[doc_id]
                    tm = val_times[doc_id]
                    times_batch[i] = tm
                    for k, v in subdoc.items():
                        data_batch[i, k] = v
                data_batch = torch.from_numpy(data_batch).float().to(device)
                times_batch = torch.from_numpy(times_batch).to(device)

                sums = data_batch.sum(1).unsqueeze(1)
                normalized_data_batch = data_batch / sums


                eta_td = eta[times_batch.type('torch.LongTensor')]
                theta, _ = self.get_theta(eta_td, normalized_data_batch, times_batch)
                alpha_td = alpha[:, times_batch.type('torch.LongTensor'), :]
                beta = self.get_beta(alpha_td).permute(1, 0, 2)
                loglik = theta.unsqueeze(2) * beta
                loglik = loglik.sum(1)
                loglik = torch.log(loglik)
                nll = -loglik * data_batch
                nll = nll.sum(-1)
                loss = nll / sums.squeeze()
                loss = loss.mean().item()
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / cnt
            ppl_all = round(math.exp(cur_loss), 1)
        return ppl_all

    




