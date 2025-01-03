# Dynamic Topic Modeling Library

This repository is derived from [the Blei lab's DETM code base](https://github.com/adjidieng/DETM) to serve as a centralized, pip-installable workspace for our use and development of this model-family in the computational humanities.  

This package can be installed from git:

```
pip install git+https://github.com/comp-int-hum/DETM.git
```

If you have your own branch, you can install that:

```
pip install git+https://github.com/comp-int-hum/DETM.git@mybranch
```

If you're working on the library itself, it's probably easiest to clone and check out your branch, and then install it in "editing mode" where you're using/testing it, so you don't need to keep reinstalling after changes:

```
git clone https://github.com/comp-int-hum/DETM.git
cd DETM
git checkout mybranch
cd /path/to/project/using/detm
pip install -e /path/back/to/DETM
```

If you have gzipped JSONL data where each line/object is a document with a text field "content" and time field "year", you can use the library somewhat like this:

```
from detm import DETM, Corpus, train_embeddings, filter_embeddings

corpus = Corpus()

with gzip.open("my_data.jsonl.gz", "rt") as ifd:
    for line in enumerate(ifd):
        corpus.append(json.loads(line)

embeddings = train_embeddings(corpus)

subdocs, times, word_list = corpus.get_filtered_subdocs(
    max_subdoc_length=500,
    content_field="content",
    time_field="year",
    min_word_count=10,
    max_word_proportion=0.7,
    lowercase=True
)

filtered_embeddings = filter_embeddings(embeddings, word_list)

model = DETM(
    num_topics=50
    min_time=min(times),
    max_time=max(times),
    window_size=25,
    embeddings=filtered_embeddings,
    device="cuda"
)

model.to(args.device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.016,
    weight_decay=1.2e-6
)

best_state = train_model(
    model=model,
    subdocs=subdocs,
    times=times,
    optimizer=optimizer,
    max_epochs=10,
    device="cuda"
)

model.load_state_dict(best_state)

test_perplexity = perplexity_on_corpus(
    model,
    some_other_corpus,
    max_subdoc_length=500,
    content_field="content",
    time_field="year",
    lowercase=True,
    device="cuda"
)


print(test_perplexity)
```

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
