"""This file defines a dynamic etm object.
"""
import math
import torch
import torch.nn.functional as F 
from torch import nn
import numpy
from torchvision.ops import MLP


class CETM_MLP(nn.Module):
    def __init__(
            self,
            num_topics,
            min_time,
            max_time,
            word_list,
            embeddings,
            enc_drop=0.0,
            
            delta=0.005,
            train_embeddings=False,
            batch_size=32,
            device="cpu",

            t_hidden_size=800, # no longer needed due to renaming
            window_size=None, # no longer needed
            
            alpha_hidden_size=800, # new
            alpha_nlayers=1, # new
            alpha_dropout=0.0, # new
            alpha_act="relu", # new

            eta_hidden_size=800,
            eta_nlayers=1,
            eta_dropout=0.0,
            eta_act="relu", # new

            theta_hidden_size=800, # new (renamed)
            theta_nlayers=1, # new
            theta_dropout=0.0, # new
            theta_act="relu",

            time_dimension=1 # dimension of the continuous input data

    ):
        super(CETM_MLP, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.word_list = word_list
        self.time_dimension = time_dimension
        
        ## define hyperparameters
        self.num_topics = num_topics
        self.max_time = max_time
        self.min_time = min_time
        
        #self.window_size = window_size
        #self.num_windows = math.ceil((max_time - min_time) / window_size)

        self.alpha_hidden_size = alpha_hidden_size
        self.alpha_nlayers = alpha_nlayers
        self.alpha_dropout = alpha_dropout
        self.alpha_act = self.get_activation(alpha_act)

        self.theta_hidden_size = theta_hidden_size
        self.theta_nlayers = theta_nlayers
        self.theta_dropout = theta_dropout
        self.theta_act = self.get_activation(theta_act)

        self.eta_hidden_size = eta_hidden_size
        self.eta_nlayers = eta_nlayers
        self.eta_dropout = eta_dropout
        self.eta_act = self.get_activation(eta_act)


        #self.t_hidden_size = t_hidden_size

        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)
        self.delta = delta
        self.train_embeddings = train_embeddings
        #self.theta_act = self.get_activation(theta_act)

        
        rho_data = numpy.array([embeddings.wv[w] for w in self.word_list])
        num_embeddings, emsize = rho_data.shape
        self.emsize = emsize
        self.rho_size = self.emsize
        
        rho = nn.Embedding(num_embeddings, self.emsize)
        rho.weight.data = torch.tensor(rho_data)
        self.rho = rho.weight.data.clone().float().to(self.device)
        
        ## define the variational parameters for the topic embeddings over time (alpha)
        self.mu_q_alpha = nn.ModuleList(
            [MLP(
                in_channels=self.time_dimension, 
                hidden_channels=[self.alpha_hidden_size] * self.alpha_nlayers + [self.rho_size], 
                dropout=self.alpha_dropout,
                ) for _ in range(self.num_topics)]
        )
        self.logsigma_q_alpha = nn.ModuleList(
            [MLP(
                in_channels=self.time_dimension, 
                hidden_channels=[self.alpha_hidden_size] * self.alpha_nlayers + [self.rho_size], 
                dropout=self.alpha_dropout,
                ) for _ in range(self.num_topics)]
        )
    
        ## define variational distribution for \theta_{1:D} via amortizartion... 
        self.mu_q_theta = MLP(
            in_channels=self.vocab_size + num_topics, 
            hidden_channels=[self.theta_hidden_size] * self.theta_nlayers + [self.num_topics], 
            dropout=self.theta_dropout,
        )
        self.logsigma_q_theta = MLP(
            in_channels=self.vocab_size + num_topics, 
            hidden_channels=[self.theta_hidden_size] * self.theta_nlayers + [self.num_topics], 
            dropout=self.theta_dropout,
        )

        ## define variational distribution for \eta via amortizartion... 
        self.mu_q_eta = MLP(
            in_channels=self.time_dimension, 
            hidden_channels=[self.eta_hidden_size] * self.eta_nlayers + [self.num_topics], 
            dropout=self.eta_dropout)
        
        self.logsigma_q_eta = MLP(
            in_channels=self.time_dimension, 
            hidden_channels=[self.eta_hidden_size] * self.eta_nlayers + [self.num_topics], 
            dropout=self.eta_dropout)

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

    def get_alpha(self, times):
        num_times = times.size(0)
        time_diff = times[1:] - times[:-1]

        alphas = torch.zeros(num_times, self.num_topics, self.rho_size, device=self.device)

        # evaluate alpha at all time points within batch
        mu_q_alpha = [self.mu_q_alpha[i](times.unsqueeze(1)) for i in range(self.num_topics)]
        logsigma_q_alpha = [self.logsigma_q_alpha[i](times.unsqueeze(1)) for i in range(self.num_topics)]
        mu_q_alpha = torch.stack(mu_q_alpha, dim=1)
        logsigma_q_alpha = torch.stack(logsigma_q_alpha, dim=1)

        # reparameterize
        alphas = self.reparameterize(mu_q_alpha, logsigma_q_alpha)

        # calculate prior distribution
        # mu_p is the previous alpha, except for the first time point, where it is 0 (from DETM code)
        # (TODO: why is this? Do we actually want the first time slice to be close to 0? ask Tom)
        mu_p = torch.cat((torch.zeros(1, self.num_topics, self.rho_size, device=self.device), alphas[:-1]), dim=0)

        # logsigma_p is the previous logsigma_q_alpha + delta * time_diff, except for the first time point, where it is 0 (sigma_p = 1)
        logsigma_p = torch.zeros_like(logsigma_q_alpha, device=self.device)
        time_diff_expanded = (self.delta * time_diff).unsqueeze(-1).unsqueeze(-1)
        logsigma_p[1:] = torch.log(torch.exp(logsigma_q_alpha[:-1]) + time_diff_expanded)

        # calculate KL divergence
        kl_alpha = self.get_kl(mu_q_alpha, logsigma_q_alpha, mu_p, logsigma_p)
        return alphas, kl_alpha.sum().sum()


    def get_eta(self, times): ## structured amortized inference
        num_times = times.size(0)
        time_diff = times[1:] - times[:-1]

        etas = torch.zeros(num_times, self.num_topics, device=self.device)
        
        mu_q = self.mu_q_eta(times.unsqueeze(1))
        logsigma_q = self.logsigma_q_eta(times.unsqueeze(1))

        etas = self.reparameterize(mu_q, logsigma_q)
        logsigma_p = torch.zeros_like(logsigma_q, device=self.device)
        time_diff_expanded = (self.delta * time_diff).unsqueeze(-1)
        logsigma_p[1:] = torch.log(torch.exp(logsigma_q[:-1]) + time_diff_expanded)
        
        mu_p = torch.cat((torch.zeros(1, self.num_topics, device=self.device), etas[:-1]), dim=0)
        kl_eta = self.get_kl(mu_q, logsigma_q, mu_p, logsigma_p)
        return etas, kl_eta.sum()
    
    def get_theta(self, eta, bows): ## amortized inference
        """Returns the topic proportions.
        """
        inp = torch.cat([bows, eta], dim=1)
        mu_theta = self.mu_q_theta(inp)
        logsigma_theta = self.logsigma_q_theta(inp)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta, torch.zeros(self.num_topics, device=self.device))
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
    
    def forward(self, bows, normalized_bows, times, num_docs):
        bows = bows.to(self.device)
        # TODO: ask tom why normalized bows are needed, and why they are normalized per batch?
        normalized_bows = normalized_bows.to(self.device)
        times = times.to(torch.float).to(self.device)

        bsz = bows.size(0)
        # TODO. aks why we are using a coefficient, is it because we want to approximate the error over the whole dataset? 
        # Does this make sense, might we be loosing alpha coherence?
        coeff = num_docs / bsz 
        alpha, kl_alpha = self.get_alpha(times)
        eta, kl_eta = self.get_eta(times)
        theta, kl_theta = self.get_theta(eta, normalized_bows)
        kl_theta = kl_theta.sum() * coeff

        beta = self.get_beta(alpha)
        nll = self.get_nll(theta, beta, bows)
        nll = nll.sum() * coeff
        nelbo = nll + kl_alpha + kl_eta + kl_theta
        return nelbo, nll, kl_alpha, kl_eta, kl_theta


    def get_completion_ppl(self, val_subdocs, val_times, device, batch_size=128):
        """Returns document completion perplexity.
        """

        self.eval()
        with torch.no_grad():
            acc_loss = 0.0
            cnt = 0
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
                times_batch = torch.from_numpy(times_batch).to(torch.float).to(device)

                sums = data_batch.sum(1).unsqueeze(1)
                normalized_data_batch = data_batch / sums

                mu_q_alpha = [self.mu_q_alpha[i](times_batch.unsqueeze(1)) for i in range(self.num_topics)]
                alphas = torch.stack(mu_q_alpha, dim=1)
                etas = self.mu_q_eta(times_batch.unsqueeze(1))

                inp = torch.cat([normalized_data_batch, etas], dim=1)
                mu_theta = self.mu_q_theta(inp)
                theta = torch.nn.functional.softmax(mu_theta, dim=-1)

                beta = self.get_beta(alphas)
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

    




