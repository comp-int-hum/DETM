import math
import logging
import re
import numpy


logger = logging.getLogger("corpus")


class Corpus(list):
    """
    
    """
        
    def _split(self, text, max_subdoc_length, lowercase):
        tokens = re.split(r"\s+", text.lower() if lowercase else text)
        num_subdocs = math.ceil(len(tokens) / max_subdoc_length)
        retval = []
        for i in range(num_subdocs):
            retval.append(tokens[i * max_subdoc_length : (i + 1) * max_subdoc_length])
        return retval

    def filter_for_model(
            self,
            model,
            max_subdoc_length,
            content_field,
            time_field=None,
            lowercase=True
    ):
        subdocs = []
        times = []
        word_to_id = {w : i for i, w in enumerate(model.word_list)}
        dropped_because_empty = 0
        dropped_because_timeless = 0
        for doc in self:
            if time_field != None:
                time = doc.get(time_field, None)
                if time != None and not numpy.isnan(time):
                    time = float(time)                    
                    #unique_times.add(time)
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
                    times.append(time)
                else:
                    dropped_because_empty += 1
        return (subdocs, times)

    def get_tokenized_subdocs(
            self,
            max_subdoc_length,
            content_field,
            lowercase=True,
    ):
        retval = []
        for doc in self:
            for subdoc_tokens in self._split(doc[content_field], max_subdoc_length, lowercase):
                retval.append(subdoc_tokens)
        return retval
        
    def get_filtered_subdocs(
            self,
            max_subdoc_length,
            content_field,
            time_field=None,
            min_word_count=1,
            max_word_proportion=1.0,
            lowercase=True,
    ):
        
        word_subdoc_count = {}
        subdoc_count = 0
        for doc in self:
            if time_field != None:
                time = doc.get(time_field, None)
                if time != None and not numpy.isnan(time):
                    time = float(time)
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

        unique_times = set()
        unique_tokens = set()
        subdocs = []
        times = []
        dropped_because_empty = 0
        dropped_because_timeless = 0
        for doc in self:
            time = None
            if time_field != None:
                time = doc.get(time_field, None)
                if time != None and not numpy.isnan(time):
                    time = float(time)                    
                    unique_times.add(time)
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
                    times.append(time)
                else:
                    dropped_because_empty += 1
        
        if time_field != None and dropped_because_timeless > 0:
            logger.info("Dropped %d documents with no time values", dropped_because_timeless)

        if dropped_because_empty > 0:
            logger.info("Dropped %d subdocs because empty or all tokens were filtered", dropped_because_empty)

        logger.info(
            "Split %d docs into %d subdocs with %d unique times and a vocabulary of %d words",
            len(self),
            len(subdocs),
            len(unique_times),
            len(word_to_id)
        )

        return (subdocs, times, [t for _, t in sorted([(i, w) for w, i in word_to_id.items()])])
