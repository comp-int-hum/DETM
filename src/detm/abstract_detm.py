from abc import ABC, abstractmethod
import math
import torch
import numpy


class AbstractDETM(torch.nn.Module, ABC):

    def __init__(self, num_topics, word_list, embeddings):
        super().__init__()
        self.word_list = word_list
        self.num_topics = num_topics
        self.device = "cpu"
        embeddings_data = numpy.array([embeddings[w] for w in self.word_list])
        num_embeddings, embeddings_size = embeddings_data.shape
        embeddings = torch.nn.Embedding(num_embeddings, embeddings_size)
        embeddings.weight.data = torch.tensor(embeddings_data)
        self.embeddings = embeddings.weight.data.clone().float()
        
    @abstractmethod
    def represent_time(self, time):
        """
        Turn a real number representing a "time" into a meaningful value for
        the model: in the original DETM, this would be an integer representing
        which window it occurs within.  For another model it might be the offset
        from a known start-time the model is/was trained to consider everything
        in relation to, etc.
        """
        pass
    
    @abstractmethod
    def topic_embeddings(self, document_times):
        """
        batch_size x num_topics x embedding_size
        """
        pass

    @abstractmethod
    def document_topic_mixture_priors(self, document_times):
        """
        batch_size x num_topics
        """
        return None

    @abstractmethod
    def document_topic_mixtures(self, document_topic_mixture_priors, document_word_counts, document_times):
        """
        batch_size x num_topics
        """
        return None
    
    def prepare_for_data(self, document_word_counts, document_times):
        pass

    def topic_distributions(self, topic_embeddings):
        tmp = topic_embeddings.view(topic_embeddings.size(0)*topic_embeddings.size(1), self.embeddings.shape[1])
        logit = torch.mm(tmp, self.embeddings.permute(1, 0)) 
        logit = logit.view(topic_embeddings.size(0), topic_embeddings.size(1), -1)
        dists = torch.nn.functional.softmax(logit, dim=-1)
        return dists
    
    def reconstruction(self, document_topic_mixtures, topic_distributions, document_word_counts):
        document_topic_mixtures = document_topic_mixtures.unsqueeze(1)
        loglik = torch.bmm(document_topic_mixtures, topic_distributions).squeeze(1)
        loglik = torch.log(loglik + 1e-6)
        loss = -loglik * document_word_counts
        loss = loss.sum(-1)
        return (None, loss)
    
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

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def forward(self, document_word_counts, document_times):
        document_word_counts = document_word_counts.to(self.device)
        document_times = document_times.to(self.device)
        document_time_representations = torch.tensor([self.represent_time(t) for t in document_times]).to(self.device)
        normalized_document_word_counts = document_word_counts / document_word_counts.sum(1).unsqueeze(1).to(self.device)
        topic_embeddings, topic_embeddings_kld = self.topic_embeddings(document_time_representations)
        document_topic_mixture_priors, document_topic_mixture_priors_kld = self.document_topic_mixture_priors(document_time_representations)
        document_topic_mixtures, document_topic_mixtures_kld = self.document_topic_mixtures(
            document_topic_mixture_priors,
            normalized_document_word_counts,
            document_time_representations
        )
        topic_distributions = self.topic_distributions(topic_embeddings)
        reconstruction, reconstruction_loss = self.reconstruction(
            document_topic_mixtures,
            topic_distributions,
            document_word_counts
        )
        nelbo = reconstruction_loss.sum() + topic_embeddings_kld.sum() + document_topic_mixture_priors_kld.sum() + document_topic_mixtures_kld.sum()
        return (nelbo, reconstruction_loss, topic_embeddings_kld, document_topic_mixture_priors_kld, document_topic_mixtures_kld)
    
    @property
    def vocab_size(self):
        return self.embeddings.shape[0]

    @property
    def embedding_size(self):
        return self.embeddings.shape[1]

    def to(self, device):
        self.embeddings = self.embeddings.to(device)
        self.device = device
        return super().to(device)    
    
