import math
import numpy
import re

class Corpus(object):
    """
    
    """
    
    def __init__(self, text_field, time_field, max_subdoc_length, lowercase=False):
        self.text_field = text_field
        self.time_field = time_field
        self.max_subdoc_length = max_subdoc_length
        self.lowercase = lowercase
        self.subdocs = []
        self.token_count = {}
        self.time_count = {}
        self.metadata = []
        self.token_in_subdoc_count = {}
        
    def add_doc(self, item, val=False):
        time = float(item[self.time_field])
        if numpy.isnan(time):
            return
        self.metadata.append({k : v for k, v in item.items() if k != self.text_field})
        for subdoc in self._split(item[self.text_field]):
            self.time_count[time] = self.time_count.get(time, 0) + 1
            for t in subdoc:
                self.token_count[t] = self.token_count.get(t, 0) + 1
            for t in set(subdoc):
                self.token_in_subdoc_count[t] = self.token_in_subdoc_count.get(t, 0) + 1
            self.subdocs.append((len(self.metadata) - 1, subdoc))

    def _split(self, text):
        tokens = self._tokenize(self._normalize(text))
        num_subdocs = math.ceil(len(tokens) / self.max_subdoc_length)
        retval = []
        for i in range(num_subdocs):
            retval.append(tokens[i * self.max_subdoc_length : (i + 1) * self.max_subdoc_length])
        return retval

    def _tokenize(self, text):
        return re.split(r"\s+", text)

    def _normalize(self, text):
        return text if isinstance(text, list) or not self.lowercase else text.lower()

    def get_vocab(self, min_count=0, max_proportion=1.0):
        vocab = {}
        for k, v in self.token_in_subdoc_count.items():
            if v >= min_count and v / len(self.subdocs) <= max_proportion:
                vocab[k] = len(vocab)
        return vocab
    
    def get_filtered_subdocs(self, vocab, window_size=None):
        if window_size:
            window_counts = self.window_counts(window_size)

        retval = []
        for i, sd in self.subdocs:
            tm = self.metadata[i][self.time_field]
            subdoc = {}
            for t in sd:
                if t in vocab:
                    subdoc[vocab[t]] = subdoc.get(vocab[t], 0) + 1
            retval.append(
                (
                    self.time2window[tm] if window_size else tm,
                    subdoc
                )
            )
        return retval

    @property
    def min_time(self):
        return min(self.time_count.keys())

    @property
    def max_time(self):
        return max(self.time_count.keys())

    @property
    def span(self):
        return self.max_time - self.min_time
    
    def window_counts(self, window_size):
        self.time2window = {}
        window_counts = {}
        cur_max_time = self.min_time
        cur_min_time = self.min_time        
        sorted_times = list(sorted(self.time_count.items()))
        for i in range(math.ceil(self.span / window_size)):
            cur_max_time += window_size
            j = 0
            while j < len(sorted_times) and sorted_times[j][0] < cur_max_time:
                self.time2window[sorted_times[j][0]] = i
                key = (cur_min_time, cur_max_time)
                window_counts[i] = window_counts.get(i, 0) + sorted_times[j][1]
                j += 1
            sorted_times = sorted_times[j:]
            cur_min_time = cur_max_time
        return window_counts
