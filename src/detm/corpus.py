import math, logging, re, numpy

logger = logging.getLogger("corpus")

class Corpus(list):
    """
    This is used for storing document corpus for DETM model
    """
    def __init__(self, documents=None):
        super().__init__(documents or [])
        
    def _split(self, text, max_subdoc_length, lowercase):
        tokens = re.split(r"\s+", text.lower() if lowercase else text)
        if max_subdoc_length == -1:
            return [tokens]
        num_subdocs = math.ceil(len(tokens) / max_subdoc_length)
        retval = []
        for i in range(num_subdocs):
            retval.append(tokens[i * max_subdoc_length : (i + 1) * max_subdoc_length])
        return retval

    def filter_corpus(
            self, max_subdoc_length, content_field,
            time_field=None, word_to_id=None,
            min_word_count=1, max_word_proportion=1.0,
            min_year = 0, max_year=9999,
            window_size=1, lowercase=True,
            logger=None):
        
        use_prexisting_word2id_flag = word_to_id is not None

        if use_prexisting_word2id_flag is False:

            word_subdoc_count = {}
            subdoc_count = 0
            for doc in self:
                if time_field != None:
                    time = doc.get(time_field, None)
                    if time != None and not numpy.isnan(time) and min_year <= int(time) < max_year:
                        time = int(time)
                    else:
                        continue # a time field was specified, but this doc has no value for it

                for subdoc_tokens in self._split(doc[content_field], max_subdoc_length, lowercase):
                    for w in set(subdoc_tokens):
                        word_subdoc_count[w] = word_subdoc_count.get(w, 0) + 1
                    subdoc_count += 1

            word_to_id = {}
            for k, v in word_subdoc_count.items():
                if v >= min_word_count and v / subdoc_count <= max_word_proportion:
                    word_to_id[k] = len(word_to_id)

        subdocs = []
        times = []
        auxiliaries = []

        # these three parameters are only used for training loop
        unique_times = set()
        dropped_because_empty = 0
        dropped_because_timeless = 0
            
        for doc in self:
            if time_field != None:
                time = doc.get(time_field, None)
                if time != None and not numpy.isnan(time):
                    time = int(time) 
                    if not (min_year <= time < max_year):
                        continue
                    window_idx = (time - min_year) // window_size
                    unique_times.add(window_idx)
                else:
                    dropped_because_timeless += 1
                    continue

            for subdoc_tokens in self._split(doc[content_field], max_subdoc_length, lowercase):
                subdoc = {}
                for t in subdoc_tokens:
                    if t in word_to_id:
                        subdoc[word_to_id[t]] = subdoc.get(word_to_id[t], 0) + 1
                if len(subdoc) > 0:
                    subdocs.append(subdoc)
                    times.append(window_idx)
                    auxiliaries.append({k: v for k, v in doc.items() 
                                        if (k != content_field)})
                    
                else:
                    dropped_because_empty += 1

        if logger:
            if time_field != None and dropped_because_timeless > 0:
                logger.info(f"Dropped {dropped_because_timeless} documents with no time values")

            if dropped_because_empty > 0:
                logger.info(f"Dropped {dropped_because_empty} subdocs because empty or all tokens were filtered")

            if use_prexisting_word2id_flag is False:
                logger.info(
                    f"Split {len(self)} docs into {len(subdocs)} subdocs with {len(unique_times)} unique times and a vocabulary of {len(word_to_id)} words")

        return (subdocs, times, auxiliaries, (None if use_prexisting_word2id_flag is True
                                                else word_to_id))