"""This file defines a dynamic etm object.
"""
import math
import logging
import torch
import numpy
from .abstract_detm import AbstractDETM
from torchvision.ops import MLP
from torch import nn


logger = logging.getLogger("xdetm")


class cETM(AbstractDETM):
    def __init__(
            self,
            num_topics,
            embeddings,
            min_time,
            max_time,
            word_list,
            window_size,

            delta=0.005,

            alpha_hidden_size=800,
            alpha_nlayers=1,
            alpha_dropout=0.0,

            eta_hidden_size=800,
            eta_nlayers=1,
            eta_dropout=0.0,

            theta_hidden_size=800,
            theta_nlayers=1,
            theta_dropout=0.0,
    ):
        super(cETM, self).__init__(num_topics, word_list, embeddings)        
        self.max_time = max_time
        self.min_time = min_time

        self.num_windows = 0
        self.num_docs = 0

        self.alpha_hidden_size = alpha_hidden_size
        self.alpha_nlayers = alpha_nlayers
        self.alpha_dropout = alpha_dropout
        self.alpha_act = torch.nn.RReLU()

        self.theta_hidden_size = theta_hidden_size
        self.theta_nlayers = theta_nlayers
        self.theta_dropout = theta_dropout
        self.theta_act = torch.nn.RReLU()

        self.eta_hidden_size = eta_hidden_size
        self.eta_nlayers = eta_nlayers
        self.eta_dropout = eta_dropout
        self.eta_act = torch.nn.RReLU()

        self.delta = delta

        self.time_dimension = 1

        ## define the variational parameters for the topic embeddings over time (alpha)
        self.mu_q_alpha = nn.ModuleList(
            [MLP(
                in_channels=self.time_dimension, 
                hidden_channels=[self.alpha_hidden_size] * self.alpha_nlayers + [self.embedding_size], 
                dropout=self.alpha_dropout,
                ) for _ in range(self.num_topics)]
        )
        self.logsigma_q_alpha = nn.ModuleList(
            [MLP(
                in_channels=self.time_dimension, 
                hidden_channels=[self.alpha_hidden_size] * self.alpha_nlayers + [self.embedding_size], 
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
        logger.info("%d windows, %d topics, %d words", self.num_windows, self.num_topics, self.vocab_size)
        
    def represent_time(self, time):
        return (time - self.min_time) / (self.max_time - self.min_time)

    def topic_embeddings(self, document_times):
        document_times = document_times.to(torch.float32)
        num_times = document_times.size(0)
        time_diff = document_times[1:] - document_times[:-1]

        alphas = torch.zeros(num_times, self.num_topics, self.embedding_size, device=self.device)

        # evaluate alpha at all time points within batch
        mu_q_alpha = [self.mu_q_alpha[i](document_times.unsqueeze(1)) for i in range(self.num_topics)]
        logsigma_q_alpha = [self.logsigma_q_alpha[i](document_times.unsqueeze(1)) for i in range(self.num_topics)]
        mu_q_alpha = torch.stack(mu_q_alpha, dim=1)
        logsigma_q_alpha = torch.stack(logsigma_q_alpha, dim=1)

        # reparameterize
        alphas = self.reparameterize(mu_q_alpha, logsigma_q_alpha)

        # calculate prior distribution
        # mu_p is the previous alpha, except for the first time point, where it is 0 (from DETM code)
        # (TODO: why is this? Do we actually want the first time slice to be close to 0? ask Tom)
        mu_p = torch.cat((torch.zeros(1, self.num_topics, self.embedding_size, device=self.device), alphas[:-1]), dim=0)

        # logsigma_p is the previous logsigma_q_alpha + delta * time_diff, except for the first time point, where it is 0 (sigma_p = 1)
        logsigma_p = torch.zeros_like(logsigma_q_alpha, device=self.device)
        time_diff_expanded = (self.delta * time_diff).unsqueeze(-1).unsqueeze(-1)
        logsigma_p[1:] = torch.log(1e-6 + time_diff_expanded)

        # calculate KL divergence
        kl_alpha = self.get_kl(mu_q_alpha, logsigma_q_alpha, mu_p, logsigma_p)
        return alphas, kl_alpha.sum()

    def document_topic_mixture_priors(self, document_times):
        document_times = document_times.to(torch.float32)
        num_times = document_times.size(0)
        time_diff = document_times[1:] - document_times[:-1]

        etas = torch.zeros(num_times, self.num_topics, device=self.device)
        
        mu_q = self.mu_q_eta(document_times.unsqueeze(1))
        logsigma_q = self.logsigma_q_eta(document_times.unsqueeze(1))

        etas = self.reparameterize(mu_q, logsigma_q)
        logsigma_p = torch.zeros_like(logsigma_q, device=self.device)
        time_diff_expanded = (self.delta * time_diff).unsqueeze(-1)
        logsigma_p[1:] = torch.log(1e-6 + time_diff_expanded)
        
        mu_p = torch.cat((torch.zeros(1, self.num_topics, device=self.device), etas[:-1]), dim=0)
        kl_eta = self.get_kl(mu_q, logsigma_q, mu_p, logsigma_p)
        return etas, kl_eta.sum()

    def document_topic_mixtures(self, document_topic_mixture_priors, document_word_counts, document_times):
        """Returns the topic proportions.
        """
        inp = torch.cat([document_word_counts, document_topic_mixture_priors], dim=1)
        mu_theta = self.mu_q_theta(inp)
        logsigma_theta = self.logsigma_q_theta(inp)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = nn.functional.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, document_topic_mixture_priors, torch.zeros(self.num_topics, device=self.device))
        return theta, kl_theta
    
    def prepare_for_data(self, document_word_counts, document_times, batch_size=1024):
        self.num_docs = len(document_word_counts)
        self.num_windows = self.num_docs
