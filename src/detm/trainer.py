import torch, math, copy, gzip
from .original import DETM

class Trainer:

    def __init__(self, logger):
        self.logger = logger
    
    def init_model(self, embeddings, word_list,
                       num_windows, num_topics, min_time, max_time,
                       t_hidden_size, eta_hidden_size,
                       enc_drop, eta_dropout, eta_nlayers, delta,
                       window_size, train_embeddings,
                       theta_act, batch_size, device):

        self.model = DETM(num_topics=num_topics, 
                          min_time=min_time, max_time=max_time,
                          embeddings=embeddings, word_list=word_list,
                          t_hidden_size=t_hidden_size,
                          eta_hidden_size=eta_hidden_size,
                          enc_drop=enc_drop, eta_dropout=eta_dropout,
                          eta_nlayers=eta_nlayers, delta=delta,
                          window_size=window_size, train_embeddings=train_embeddings,
                          num_windows=num_windows, theta_act=theta_act, 
                          batch_size=batch_size, device=device)
        self.model.to(device)
    
    def init_training_params(self, num_train, num_eval,
                             learning_rate, wdecay, clip,
                             reduce_rate, lr_factor, early_stop
                             ):

        assert self.model

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=wdecay
            )
        
        self.clip, self.num_train, self.num_eval = clip, num_train, num_eval
        self.reduce_rate, self.lr_factor, self.early_stop = reduce_rate, lr_factor, early_stop
        self.epoch, self.since_improvement, self.since_annealing = 0, 0, 0
        self.best_state, self.best_eval_ppl = None, None
    
    def load_model(self, model_dir, device):
        with gzip.open(model_dir, "rb") as ifd:
            self.model = torch.load(ifd, map_location=torch.device(device))
        
        return {token: idx for idx, token in enumerate(self.model.word_list)}
    
    def start_epoch(self):
        self.logger.info(f"Starting epoch {self.epoch}")
        self.train_acc_nelbo = 0
        self.train_acc_nll = 0
        self.train_acc_kl_alpha_loss = 0
        self.train_acc_kl_eta_loss = 0
        self.train_acc_kl_theta_loss = 0
        self.train_cnt = 0

        self.eval_acc_loss = 0
        self.eval_cnt = 0
    
    def train_model(self, batch_generator, rnn_input):
        self.model.train()

        try:
            while True:  
                data_batch, normalized_data_batch, times_batch, _ = next(batch_generator)
                nelbo, nll, kl_alpha, kl_eta, kl_theta = self.model(data_batch, normalized_data_batch, 
                                                                    times_batch, rnn_input, 
                                                                    self.num_train)
                curr_nelbo = torch.sum(nelbo).item()
                curr_nll = torch.sum(nll).item()
                curr_kl_alpha = torch.sum(kl_alpha).item()
                curr_kl_eta = torch.sum(kl_eta).item()
                curr_kl_theta = torch.sum(kl_theta).item()
                self.logger.info(
                "KL_theta: {}, KL_eta: {}, KL_alpha: {}, Rec_loss: {}, NELBO: {}".format(
                curr_kl_theta, curr_kl_eta, curr_kl_alpha, curr_nll, curr_nelbo
                ))
                
                self.train_acc_nelbo += curr_nelbo
                self.train_acc_nll += curr_nll
                self.train_acc_kl_alpha_loss += curr_kl_alpha
                self.train_acc_kl_eta_loss += curr_kl_eta
                self.train_acc_kl_theta_loss += curr_kl_theta
                self.train_cnt += 1

                if not torch.any(torch.isnan(nelbo)):
                    nelbo.backward()
                    if self.clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()

        except StopIteration:
            pass
    
    def eval_model(self, batch_generator, rnn_input):
        self.model.eval()
        with torch.no_grad():
            try:
                while True:
                    data_batch, normalized_data_batch, times_batch, _ = next(batch_generator)
                    loss = self.model(data_batch, normalized_data_batch, 
                                      times_batch, rnn_input, 
                                      self.num_eval, training=False)
                    
                    self.logger.info(f"current batch eval loss {loss}, count {self.eval_cnt}")
                    self.eval_acc_loss += loss
                    self.eval_cnt += 1
                    
            except StopIteration:
                pass
    
    def apply_model(self, dataloader, num_appl, window_num, seed):
        vocab_num = len(self.model.word_list)
        batch_generator = dataloader.batch_generator(vocab_num, seed, self.logger)
        rnn_input = dataloader.get_rnn(window_num, vocab_num)
        self.model.eval()
        with torch.no_grad():
            try:
                while True:
                    data_batch, normalized_data_batch, times_batch, data_inds = next(batch_generator)
                    liks = self.model(data_batch, normalized_data_batch, 
                                      times_batch, rnn_input, num_appl, 
                                      training=False, get_lik=True)
                    
                    dataloader.update_subdoc_counts(liks, data_inds, self.model.word_list, self.logger)
                    
            except StopIteration:
                pass
        
        return dataloader.get_appl_data()
    
    def end_epoch(self) -> bool:
        if math.isnan(self.eval_acc_loss) or self.eval_cnt <= 0:
            eval_ppl = float('nan')
        else:
            eval_ppl = round(math.exp(self.eval_acc_loss / self.eval_cnt))
        
        self.logger.info(
            "Epoch {}: LR: {}, KL_theta: {}, KL_eta: {}, KL_alpha: {}, Rec_loss: {}, NELBO: {}, PPL: {}".format(
                self.epoch, self.optimizer.param_groups[0]["lr"],
                round(self.train_acc_kl_theta_loss / self.train_cnt, 2),
                round(self.train_acc_kl_eta_loss / self.train_cnt, 2),
                round(self.train_acc_kl_alpha_loss / self.train_cnt, 2),
                round(self.train_acc_nll / self.train_cnt, 2),
                round(self.train_acc_nelbo / self.train_cnt, 2),
                round(eval_ppl, 1)
            )
        )

        self.epoch += 1
    
        if not self.best_eval_ppl or eval_ppl < self.best_eval_ppl:
            self.logger.info("Copying new best model...")
            self.best_eval_ppl = eval_ppl
            self.best_state = copy.deepcopy(self.model.state_dict())
            self.since_improvement = 0
            self.logger.info("Copied.")
        else:
            self.since_improvement += 1
        self.since_annealing += 1
        if (
            self.since_improvement > self.reduce_rate and 
            self.since_annealing > self.reduce_rate
        ):
            self.optimizer.param_groups[0]["lr"] /= self.lr_factor
            self.model.load_state_dict(self.best_state)
            self.since_annealing = 0
        elif self.since_improvement >= self.early_stop:
            return False
        
        return True

    def get_best_model(self):
        self.model.load_state_dict(self.best_state)
        return self.model