# Dynamic Topic Modeling Library

This repository is derived from [the Blei lab's DETM code base](https://github.com/adjidieng/DETM) to serve as a centralized, pip-installable workspace for our use and development of this model-family in the computational humanities.  

This package can be installed from git:

```bash
pip install git+https://github.com/comp-int-hum/DETM.git
```

If you have your own branch, you can install that:

```bash
pip install git+https://github.com/comp-int-hum/DETM.git@mybranch
```

If you're working on the library itself, it's probably easiest to clone and check out your branch, and then install it in "editing mode" where you're using/testing it, so you don't need to keep reinstalling after changes:

```bash
git clone https://github.com/comp-int-hum/DETM.git
cd DETM
git checkout mybranch
cd /path/to/project/using/detm
pip install -e /path/back/to/DETM
```

If you have gzipped JSONL data where each line/object is a document with a text field "content" and time field "year", you can use the library somewhat like this:

```python
import torch
from detm import xDETM, Corpus, train_embeddings, apply_model, train_model

corpus = Corpus()

with gzip.open("my_data.jsonl.gz", "rt") as ifd:
    for line in enumerate(ifd):
        corpus.append(json.loads(line)

embeddings = train_embeddings(corpus, content_field="content", max_subdoc_length=500, lowercase=True, random_seed=42)

subdocs, times, word_list = corpus.get_filtered_subdocs(
    max_subdoc_length=100,
    content_field="content",
    time_field="year",
    min_word_count=10,
    max_word_proportion=0.7,
    lowercase=True
)

model = xDETM(
    word_list=word_list,
    num_topics=50
    min_time=min(times),
    max_time=max(times),
    window_size=50,
    embeddings=embeddings
)

model.to("cuda")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.004,
    weight_decay=1.2e-6
)

best_state = train_model(
    model=model,
    subdocs=subdocs,
    times=times,
    optimizer=optimizer,
    max_epochs=10,
    batch_size=1024,
    device="cuda"
)

model.load_state_dict(best_state)
```

Note that training stability is sensitive to learning rate, which needs to be smaller as model size (topics, vocabulary, etc) increases: if you see NaN errors, try lowering the learning rate from the start (TODO: look into a reasonable formula for experimentally tuning this in the first epoch).

Assuming you have loaded some other data to get different subdocs/times, you could then run:

```
test_labels, test_perplexity = apply_model(
    model,
    other_subdocs,
    other_times,
    batch_size=1024,
    device="cuda"
)

print(test_perplexity)
```

Of course, with an unsupervised model, it may make sense to simply apply it back to the training data.

Everything up to the point of our initial fork should be attributed to:

```
@article{dieng2019dynamic,
  title={The Dynamic Embedded Topic Model},
  author={Dieng, Adji B and Ruiz, Francisco JR and Blei, David M},
  journal={arXiv preprint arXiv:1907.05545},
  year={2019}
}
```

See the corresponding [README](README.original.md) for more information on the original repository.
