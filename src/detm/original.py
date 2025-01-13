"""This file defines a dynamic etm object.
"""
import math, torch
import torch.nn.functional as F 
from torch import nn

class DETM(nn.Module):
    def __init__(
            self, num_topics, 
            min_time, max_time,
            embeddings, word_list,
            t_hidden_size=800,
            eta_hidden_size=200,
            enc_drop=0.0, eta_dropout=0.0,
            eta_nlayers=3, delta=0.005,
            window_size=None, train_embeddings=False,
            num_windows=None,
            theta_act="relu", batch_size=32,
            device="cpu"
    ):
        super(DETM, self).__init__()

        self.device = device
        self.batch_size = batch_size
        
        ## define hyperparameters
        self.num_topics = num_topics
        self.max_time = max_time
        self.min_time = min_time
        self.window_size = window_size
        self.num_windows = (num_windows 
                            if num_windows 
                            else math.ceil((max_time - min_time) / window_size))
        self.t_hidden_size = t_hidden_size
        self.eta_hidden_size = eta_hidden_size

        self.enc_drop = enc_drop
        self.eta_nlayers = eta_nlayers
        self.t_drop = nn.Dropout(enc_drop)
        self.delta = delta
        self.train_embeddings = train_embeddings
        self.theta_act = self.get_activation(theta_act)

        self.word_list = word_list
        rho_data = embeddings
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
        
        if not self.training:
            return self.mu_q_alpha, None
        
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
    
    def get_beta(self, alpha, times):
        """Returns the topic matrix \beta of shape K x V
        """
        if not self.training:
            alpha = alpha[:, times.type("torch.LongTensor"), :]
        tmp = alpha.view(alpha.size(0)*alpha.size(1), self.rho_size)
        logit = torch.mm(tmp, self.rho.permute(1, 0)) 
        logit = logit.view(alpha.size(0), alpha.size(1), -1)
        beta = F.softmax(logit, dim=-1)
        return beta[times.type('torch.LongTensor')] if self.training else beta.permute(1, 0, 2)

    def get_nll(self, theta, beta, bows):
        theta = theta.unsqueeze(1)
        lik = torch.bmm(theta, beta).squeeze(1)
        loglik = torch.log(lik + 1e-6)
        nll = -loglik * bows
        nll = nll.sum(-1)
        return nll

    def get_lik(self, theta, beta):
        lik = theta.unsqueeze(2) * beta
        return lik
    
    def forward(self, bows, normalized_bows, times, rnn_inp, num_docs,
                training=True, get_lik=False):
        self.training = training
        
        # if it is get likelihood, it should not be in the training loop
        assert (get_lik is False) or (get_lik is True and training is False)
        
        bows = bows.to(self.device)
        normalized_bows = normalized_bows.to(self.device)
        times = times.to(self.device)
        rnn_inp = rnn_inp.to(self.device)
        bsz = normalized_bows.size(0)
        coeff = num_docs / bsz 

        alpha, kl_alpha = self.get_alpha()
        eta, kl_eta = self.get_eta(rnn_inp)
        theta, kl_theta = self.get_theta(eta, normalized_bows, times)
        beta = self.get_beta(alpha, times)

        if get_lik:
            return self.get_lik(theta, beta)
        
        nll = self.get_nll(theta, beta, normalized_bows)

        if self.training:
            nll = nll.sum() / bsz
            # * coeff
            kl_theta = kl_theta.sum() / bsz
            # * coeff
            nelbo = nll + kl_alpha + kl_eta + kl_theta
            return nelbo, nll, kl_alpha, kl_eta, kl_theta
        else:
            # sums = bows.sum(dim=1, keepdim=True)
            # loss = nll / sums.squeeze()
            loss = nll.mean().item()
            return loss

    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for \eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        return (weight.new_zeros(nlayers, 1, nhid), 
                weight.new_zeros(nlayers, 1, nhid))