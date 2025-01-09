import math, json
from collections import Counter
import numpy as np

from .corpus import Corpus
from .utils import open_jsonl_file

class Dataset:

    def __init__(self, train_dir, eval_dir=None, train_proportion=0.8):
        self.train_dir = train_dir
        self.eval_dir = eval_dir
        self.train_proportion = train_proportion
    
    def get_data(self, is_train=True):
        if is_train:
            return self.t_subdocs, self.t_times, self.t_auxiliaries, self.t_time_counter
        
        return self.e_subdocs, self.e_times, self.e_auxiliaries
    
    def preprocess_data(self, min_time, max_time, window_size,
                        content_field, time_field,
                        max_subdoc_length=500,
                        min_word_occurrance=0, max_word_proportion=1.0,
                        logger=None, word2id=None):
        
        if logger:
            logger.info(f"----- starts preprocessing data ----- ")

        train_corpus = Corpus([])
        eval_corpus = Corpus([])

        train_data_generator = open_jsonl_file(self.train_dir)
        try:
            while True:
                entry = next(train_data_generator)
                train_corpus.append(json.loads(entry))
        except StopIteration:
            pass
        
        if word2id:
            t_subdocs, t_times, t_auxiliaries, _ = train_corpus.filter_corpus(
                max_subdoc_length,
                content_field, time_field,
                word_to_id=word2id,
                min_year=min_time, max_year=max_time,
                window_size=window_size,
                logger=logger)
            
            if self.train_proportion == -1:
                self.e_subdocs = t_subdocs
                self.e_times = t_times
                self.e_auxiliaries = t_auxiliaries
                return None
        else:
            t_subdocs, t_times, t_auxiliaries, word2id = train_corpus.filter_corpus(
                max_subdoc_length,
                content_field, time_field,
                min_word_count=min_word_occurrance, max_word_proportion=max_word_proportion,
                min_year=min_time, max_year=max_time, window_size=window_size,
                logger=logger)
        
        if self.eval_dir:
            eval_data_generator = open_jsonl_file(self.eval_dir)
            try:
                while True:
                    entry = next(eval_data_generator)
                    eval_corpus.append(json.loads(entry))
            except StopIteration:
                pass
            e_subdocs, e_times, e_auxiliaries, _ = eval_corpus.filter_corpus(
                max_subdoc_length, content_field, time_field, word_to_id=word2id,
                min_word_count=min_word_occurrance, max_word_proportion=max_word_proportion,
                min_year=min_time, max_year=max_time, window_size=window_size,
                logger=logger
            )

        else:
            t_subdocs, t_times, t_auxiliaries, e_subdocs, e_times, e_auxiliaries = self.split_train_eval(
                t_subdocs, t_times, t_auxiliaries, self.train_proportion, logger=logger)
        
        t_subdocs, t_times, t_auxiliaries, time_counter = self.organize_train_data_by_times(t_subdocs, t_times, t_auxiliaries, logger=logger)

        self.t_subdocs = t_subdocs
        self.t_times = t_times
        self.t_time_counter = time_counter
        self.t_auxiliaries = t_auxiliaries

        self.e_subdocs = e_subdocs
        self.e_times = e_times
        self.e_auxiliaries = e_auxiliaries

        word2id = sorted(word2id.items(), key=lambda x : x[1])
        return [w[0] for w in word2id]
    
    def split_train_eval(self, subdocs, times, auxiliaries, train_proportion, logger):
        
        if logger:
            logger.info(f"----- starts splitting train and eval ----- ")

        assert len(subdocs) == len(times)
        assert len(times) == len(auxiliaries)
        shuffled_indices = np.random.permutation(len(times))
        train_eval_split = math.ceil(len(times) * train_proportion)

        train_indices = set(shuffled_indices[:train_eval_split])
        eval_indices = set(shuffled_indices[train_eval_split:])
        t_subdocs, e_subdocs = [], []
        t_times, e_times = [], []
        t_auxiliaries, e_auxiliaries = [], []

        for idx, (subdoc, time, auxiliary) in enumerate(
            zip(subdocs, times, auxiliaries)
            ):
            if idx in train_indices:
                t_subdocs.append(subdoc)
                t_times.append(time)
                t_auxiliaries.append(auxiliary)
            elif idx in eval_indices:
                e_subdocs.append(subdoc)
                e_times.append(time)
                e_auxiliaries.append(auxiliary)
        
        t_counter = Counter(t_times)
        e_counter = Counter(e_times)

        if logger:
            logger.info(f"training time counter: {t_counter};\n eval time counter: {e_counter}")
            logger.info(f"----- complete splitting train and eval ----- ")

        return t_subdocs, t_times, t_auxiliaries, e_subdocs, e_times, e_auxiliaries

    def organize_train_data_by_times(self, subdocs, times, auxiliaries, logger):

        if logger:
            logger.info(f"----- starts organzing data by time ----- ")

        sorted_indices = sorted(range(len(times)), key=lambda i: times[i])

        subdocs = [subdocs[i] for i in sorted_indices]
        times = [times[i] for i in sorted_indices]
        auxiliaries = [auxiliaries[i] for i in sorted_indices]
    
        time_counter = sorted(Counter(times).items(), 
                              key=lambda x: x[0])
        buffer = 0
        for time_idx, counts in time_counter:
            time_set = set(times[buffer:buffer+counts])
            try:
                assert len(time_set) == 1 and list(time_set)[0] == time_idx

                if logger:
                    logger.info(f"expected {counts} in time idx {time_idx}, confirmed")
            except AssertionError:
                raise AssertionError(f"expected {counts} in time idx {time_idx}, but instead got {times[buffer:buffer+counts]}")
            buffer += counts

        if logger:
            logger.info(f"----- completes organizing data by time ----- ")
        return subdocs, times, auxiliaries, time_counter