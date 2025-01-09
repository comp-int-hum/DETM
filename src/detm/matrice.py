from .trainer import Trainer
from .data import Dataset
from .dataloader import DataLoader
from .utils import get_window_ranges, translate_text_deepl

import numpy as np
from typing import List, Dict

class DETM_Matrice:

    def __init__(self, model_dir, input_dir):
        self.model_dir = model_dir
        self.input_dir = input_dir
        pass

    def get_matrice(self, device, min_time, max_time, window_size,
                    num_topics, content_field, time_field,
                    batch_size, max_subdoc_length, 
                    min_word_occurrence, max_word_proportion,
                    logger, random_seed):
        
        if logger:
            logger.info(f"----- starts getting matrice ------ ")
        
        docs, doc2title, doc2author, doc2year = {}, {}, {}, {}
        unique_authors = set()
        
        subdocs_data = self._apply_model(device, min_time, max_time, window_size,
                                        content_field, time_field,
                                        batch_size, max_subdoc_length, 
                                        min_word_occurrence, max_word_proportion,
                                        logger, random_seed)
        
        for subdoc in subdocs_data:
            title = subdoc["title"]
            author = subdoc["author_info"]
            year = subdoc["written_year"]
            htid = subdoc["htid"]
            doc2title[htid] = title
            doc2author[htid] = author
            doc2year[htid] = year
            docs.setdefault(htid, [])
            docs[htid].append(subdoc)
            unique_authors.add(author)

        window_rangs = get_window_ranges(min_time, max_time, window_size, get_str=False)
        nwins, nwords, ntopics, nauths, ndocs = (len(window_rangs), len(self.token2id), 
                                                 num_topics, len(unique_authors), len(docs))
       
        token2id = self.token2id
        author2id = {a : i for i, a in enumerate(unique_authors)}
        htid2id = {d : i for i, d in enumerate(docs.keys())}
        time2window = {}
        for idx, years in enumerate(window_rangs):
            time2window.update({year: idx for year in years})
    
        if logger:
            logger.info(
                f"Found {nwins} windows, {nwords} unique words, {ntopics} unique topics, " +
                f"{ndocs} unique documents, and {nauths} unique authors"
            )

        words_wins_topics = np.zeros(shape=(nwords, nwins, ntopics))
        auths_wins_topics = np.zeros(shape=(nauths, nwins, ntopics))
        htid_wins_topics = np.zeros(shape=(ndocs, nwins, ntopics))
    
        for htid, subdocs in docs.items():
            title = doc2title[htid]
            author = doc2author[htid]
            year = doc2year[htid]
            aid = author2id[author]
            win = time2window[year]
            did = htid2id[htid]
        
            for subdoc in subdocs:
                for word, topic in subdoc["tokens"]:
                    if topic != None:
                        wid = token2id[word]
                        words_wins_topics[wid, win, topic] += 1
                        auths_wins_topics[aid, win, topic] += 1
                        htid_wins_topics[did, win, topic] += 1
        
        self.start_time = min_time
        self.id2author = {i : a for a, i in author2id.items()}
        self.id2word = {i : w for w, i in token2id.items()}
        self.id2htid = {i : d for d, i in htid2id.items()}
        self.doc2title = doc2title
        self.doc2author = doc2author
        self.doc2year = doc2year
        self.wwt = words_wins_topics
        self.awt = auths_wins_topics
        self.hwt = htid_wins_topics

        if logger:
            logger.info(f"----- completes getting matrice ------ ")
    
    def _apply_model(self, device,
                    min_time, max_time, window_size,
                    content_field, time_field, batch_size,
                    max_subdoc_length, min_word_occurrence,
                    max_word_proportion, logger=None,
                    random_seed=42
                    ):

        if logger:
            logger.info(f"----- starts applying model ------ ")

        trainer: Trainer = Trainer(logger)
        self.token2id: Dict[str, int] = trainer.load_model(self.model_dir, device)
        self.all_window_ranges: List[str] = get_window_ranges(min_time, max_time, window_size)
    
        dataset: Dataset = Dataset(self.input_dir, None, -1)
        dataset.preprocess_data(min_time, max_time, window_size,
                            content_field, time_field,
                            max_subdoc_length=max_subdoc_length, 
                            min_word_occurrance=min_word_occurrence, 
                            max_word_proportion=max_word_proportion,
                            logger=logger, word2id=self.token2id)

        a_subdocs, a_times, a_auxiliaries = dataset.get_data(is_train=False)
        num_appl: int = len(a_times)
        appl_dataloader: DataLoader = DataLoader(a_subdocs, a_times, a_auxiliaries, 
                                                 batch_size, device, self.all_window_ranges)

        if logger:
            logger.info(f"current have {len(self.all_window_ranges)} time windows"
                        + f" and {num_appl} application instances")                                                                       
    
        del dataset, a_subdocs, a_times, a_auxiliaries

        subdoc_data = trainer.apply_model(appl_dataloader, len(self.all_window_ranges), 
                                            num_appl, random_seed)

        if logger:
            logger.info(f"----- completes applying model ------ ")
        
        return subdoc_data
    
    def get_top_work_for_topic(self, top_n=8, epsilon=1e-10, translate=False):
    
        # htid_win_topic has original shape (num_htid, num_time, num_topics)

        # 1. distribution of the topic per work (e.g. how is the topic composed of via each topic)
        # htid_win_topic.sum(2) has shape: (num_htid, num_times) -> sum of score over each htid

        # 2. proportion of the work in single topic single window
        #  (e.g. how much of each work is each topic per window constituted of)
        # htid_win_topic.sum(0) has shape: (num_times, num_topics) -> sum of score over each topic

        return self._get_prop_and_dist_helper(self.hwt, self.id2htid, top_n, epsilon, translate)
    
    def get_top_vocab_for_topic(self, top_n=8, epsilon=1e-10, translate=False):
        
        return self._get_prop_and_dist_helper(self.wwt, self.id2word, top_n, epsilon, translate)
    
    def get_top_author_for_topic(self, top_n=8, epsilon=1e-10, translate=False):

        return self._get_prop_and_dist_helper(self.awt, self.id2author, top_n, epsilon, translate)

    def _get_prop_and_dist_helper(self, data_window_topic, id2data, top_n, epsilon,
                                  translate):

        return_data = {}

        data_dist_per_top = (data_window_topic.transpose(2, 0, 1) / 
                                   (data_window_topic.sum(2) + epsilon)).transpose(1, 2, 0)

        topic_dist_per_data = data_window_topic / (data_window_topic.sum(0) + epsilon)

        const_str = "Data proportion in single topic "

        for topic_idx in range(20):
            for temp_idx in range(5):
                score_per_topic_per_time = data_dist_per_top[:, temp_idx, topic_idx]
                top_n_indices = np.argsort(score_per_topic_per_time)[-top_n:][::-1]
                top_n_data = score_per_topic_per_time[top_n_indices]
                key = const_str + f"#{topic_idx} time {self.all_window_ranges[temp_idx]}"
                return_data[key] = {
                    f"# {idx}": {'score': round(top_n_data[idx], 4), 
                                 'name': (translate_text_deepl(id2data[top_n_indices[idx]]) 
                                          if translate 
                                          else id2data[top_n_indices[idx]])}
                    for idx in range(top_n)}
        
        # only per topic
        for topic_idx in range(20):
            # summing over the time axis
            score_per_topic = (data_dist_per_top.sum(1))[:, topic_idx]
            top_n_indices = np.argsort(score_per_topic)[-top_n:][::-1]
            top_n_data = score_per_topic[top_n_indices]
            key = const_str + f"#{topic_idx} across all window"
            return_data[key] = {
                    f"# {idx}": {'score': round(top_n_data[idx], 4), 
                                 'name': (translate_text_deepl(id2data[top_n_indices[idx]]) 
                                          if translate 
                                          else id2data[top_n_indices[idx]])}
                    for idx in range(top_n)}
        
        const_str = "Topic proportion in single data "

        for topic_idx in range(20):
            for temp_idx in range(5):
                score_per_topic_per_time = topic_dist_per_data[:, temp_idx, topic_idx]
                top_n_indices = np.argsort(score_per_topic_per_time)[-top_n:][::-1]
                top_n_data = score_per_topic_per_time[top_n_indices]
                key = const_str + f"#{topic_idx} time {self.all_window_ranges[temp_idx]}"
                return_data[key] = {
                    f"# {idx}": {'score': round(top_n_data[idx], 4), 
                                 'name': (translate_text_deepl(id2data[top_n_indices[idx]]) 
                                          if translate 
                                          else id2data[top_n_indices[idx]])}
                    for idx in range(top_n)}
        
        # only per topic
        for topic_idx in range(20):
            # summing over the time axis
            score_per_topic = (topic_dist_per_data.sum(1))[:, topic_idx]
            top_n_indices = np.argsort(score_per_topic)[-top_n:][::-1]
            top_n_data = score_per_topic[top_n_indices]
            key = const_str + f"#{topic_idx} across all window"
            return_data[key] = {
                    f"# {idx}": {'score': round(top_n_data[idx], 4), 
                                 'name': (translate_text_deepl(id2data[top_n_indices[idx]]) 
                                          if translate 
                                          else id2data[top_n_indices[idx]])}
                    for idx in range(top_n)}
        
        return return_data