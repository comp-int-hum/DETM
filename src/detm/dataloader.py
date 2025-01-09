import math, random, torch
from collections import Counter
import numpy as np

class DataLoader:
    
    def __init__(self, subdocs, times, auxiliaries,
                 batch_size, device,
                 all_window_ranges,
                 time_counter = None):
        
        self.subdocs = subdocs
        self.times = times
        self.auxiliaries = auxiliaries
        self.data_length = len(times)
        self.batch_size = batch_size
        self.device = device
        self.all_window_ranges = all_window_ranges
        self.time_counter = time_counter

        if self.time_counter and len(self.time_counter) > self.batch_size:
            raise Exception("unable to form batch instances because there are more time windows than single batch size")

    def get_rnn(self, num_window, num_vocab):

        indices = torch.arange(0, self.data_length, dtype=torch.int)
        indices = torch.split(indices, self.batch_size)
        rnn_input = torch.zeros(num_window, num_vocab).to(self.device)
        cnt = torch.zeros(num_window).to(self.device)

        for _, ind in enumerate(indices):
            batch_size = len(ind)
            data_batch = np.zeros((batch_size, num_vocab))
            times_batch = np.zeros((batch_size,))
            for i, doc_id in enumerate(ind):
                times_batch[i] = self.times[doc_id]  # timestamp
                for k, v in self.subdocs[doc_id].items():
                    data_batch[i, k] = v

            data_batch = torch.from_numpy(data_batch).float().to(self.device)
            times_batch = torch.from_numpy(times_batch).to(self.device)

            for t in range(num_window):
                tmp = (times_batch == t).nonzero()
                docs = data_batch[tmp].squeeze().sum(0)
                rnn_input[t] += docs
                cnt[t] += len(tmp)
        rnn_input = rnn_input / cnt.unsqueeze(1)
        return rnn_input
    
    def _get_split_indices(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

        if self.time_counter:
        
            total_batch_num = math.ceil(self.data_length / self.batch_size)
            batch_indices_list = [[] for _ in range(total_batch_num)]
            buffer = 0

            total_used_idx = set()
            for idx, (_, curr_count) in enumerate(self.time_counter):
                if curr_count == 0:
                    continue
              
                prev_section = buffer
                buffer += curr_count
                cur_section = buffer

                all_indices = torch.arange(prev_section, cur_section)
                division = max(1, total_batch_num // len(all_indices))
                all_indices = all_indices.repeat(division)
                num_sample = max(0, total_batch_num - len(all_indices))
                sampled_indices = torch.randperm(len(all_indices))[:num_sample]
                all_indices = torch.cat([all_indices, sampled_indices])
                permuted_indices = all_indices[torch.randperm(all_indices.size(0))]
                permuted_indices = permuted_indices[:total_batch_num]
                    
                for idx in range(len(permuted_indices)):
                    batch_indices_list[idx].append(permuted_indices[idx])
                
                total_used_idx.update(permuted_indices)
      
            remaining_indices = [idx for idx in range(self.data_length) if idx not in total_used_idx]
            remaining_indices = torch.tensor(remaining_indices)[torch.randperm(len(remaining_indices))]
            ptr = 0

            for idx in range(len(batch_indices_list)):
                num_missing_items = self.batch_size - len(batch_indices_list[idx])
                batch_indices_list[idx].extend(remaining_indices[ptr:ptr+num_missing_items])
                ptr += num_missing_items

        else:
            all_batch_indices = torch.randperm(self.data_length)
            batch_indices_list = torch.split(all_batch_indices, self.batch_size) 
        return batch_indices_list

    def batch_generator(self, vocab_size, seed, logger=None):

        if logger and self.time_counter:
            buffer = 0
            for time_idx, counts in self.time_counter:
                time_set = set(self.times[buffer:buffer+counts])
                try:
                    assert len(time_set) == 1 and list(time_set)[0] == time_idx
                    # logger.info(f"expected {counts} in time idx {time_idx}, confirmed")
                except AssertionError:
                    raise AssertionError(f"expected {counts} in time idx {time_idx}, but instead got {self.times[buffer:buffer+counts]}")
                buffer += counts

        batch_indices_list = self._get_split_indices(seed)

        all_missing_time_counter = Counter()

        for idx, batch_indices in enumerate(batch_indices_list):
            current_batch_size = len(batch_indices)
            data_batch = torch.zeros((current_batch_size, vocab_size), device=self.device)
            times_batch = torch.zeros((current_batch_size,), device=self.device)
            for i, doc_id in enumerate(batch_indices):
                times_batch[i] = self.times[doc_id]
                for k, v in self.subdocs[doc_id].items():
                    data_batch[i, k] = v

            if logger:
                time_included = Counter(times_batch.tolist())
                # logger.info(f"{time_included}")
                time_included = time_included.keys()
                time_missed = [window_range for (idx, window_range) in 
                               enumerate(self.all_window_ranges) if idx not in time_included]
                all_missing_time_counter.update(time_missed)

                if idx == len(batch_indices_list) - 1:
                    logger.info(f"all missing time in the current counter: {all_missing_time_counter}")
            
            sums = data_batch.sum(dim=1, keepdim=True)
            sums[sums == 0] = 1
            normalized_data_batch = data_batch / sums

            yield data_batch, normalized_data_batch, times_batch, batch_indices

    def update_subdoc_counts(self, liks, inds, token_list, logger=None):
        if not hasattr(self, 'subdoc_topics') or not self.subdoc_topics:
            self.subdoc_topics = [[(token_list[idx], idx) for idx, _ in subdoc.items()] for subdoc in self.subdocs]
        
        try:
            for lik, ind in zip(liks, inds):
                lik = lik.argmax(0)
                self.subdoc_topics[ind] = [
                    (tok, lik[idx].item())
                    for tok, idx in self.subdoc_topics[ind]
                    ]
                
        except Exception as e:
            if logger:
                logger.info(f"received error when enumerating text : {str(e)}")
            raise Exception(e)
    
    def get_appl_data(self):
        return_arr = []
        assert len(self.subdoc_topics) == len(self.auxiliaries)
        for idx, auxiliary in enumerate(self.auxiliaries):
            auxiliary['tokens'] = self.subdoc_topics[idx]
            return_arr.append(auxiliary)
        
        return return_arr