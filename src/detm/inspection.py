import unicodedata
import re
from gensim.models.keyedvectors import KeyedVectors


def get_nearest_neighbors(model, words):
    kv = KeyedVectors(300, count=len(model.all_embeddings))
    kv.add_vectors([x for x, _ in model.all_embeddings], [x for _, x in model.all_embeddings])
    retval = {}
    for word in words:
        retval[word] = kv.most_similar(word)
    return retval


def get_topics(model):
    kv = KeyedVectors(300, count=len(model.all_embeddings))
    kv.add_vectors([x for x, _ in model.all_embeddings], [x for _, x in model.all_embeddings])
    alpha = model.get_alpha()[0].cpu().detach().numpy()
    kv.similar_by_vector(alpha[0,0,:])


cache = {}
def normalize(w):
    return w
    chars = []
    for c in w:
        cache[c] = cache.get(c, unicodedata.lookup(re.sub(r" WITH .*$", "", unicodedata.name(c))))
        chars.append(cache[c])
    return "".join(chars).lower()


def preprocess(cube, annotations):

    topic_table, word_table, document_table, window_table = [], [], [], []
    topic_table = {
        "ids" : [],
        "top_words" : [],
        "metrics" : {
            "Entropy" : []
        },
        "annotations" : {
        },
        "window_weights" : []
    }
    
    words = numpy.array([normalize(cube["index_to_word"][i].lower()) for i in range(len(cube["index_to_word"]))])
    stems = {}
    stem_to_word_indices = {}
    for index, word in cube["index_to_word"].items():
        nw = normalize(word)
        #if args.stop and ((not args.language and nw in default_stops) or nw in stops.get(args.language, [])):
        #    continue
        #stem = word if not args.stem else default_stemmer.stem(nw) if args.language not in stemmers else
        #stem = stemmers[args.language].lemmatize([nw])[0][1]
        stem_to_word_indices[nw] = stem_to_word_indices.get(nw, [])
        stem_to_word_indices[nw].append(index)
    stem_to_index = {stem : i for i, stem in enumerate(stem_to_word_indices.keys())}
    index_to_stem = {i : stem for stem, i in stem_to_index.items()}

    stems = numpy.array([index_to_stem[i] for i in range(len(index_to_stem))])

    
    P_wbt = torch.from_numpy(numpy.transpose(cube["topic_window_word"], (2, 1, 0))) # P(w|b,t), word x bucket x topic (.sum(0) == 1.0)
        

    P_tbw = torch.permute(P_wbt / torch.unsqueeze(P_wbt.sum(2), 2), (2, 1, 0)) # topic x bucket x word
    P_tbs = torch.zeros(size=(P_tbw.shape[0], P_tbw.shape[1], len(stem_to_index)))
    P_sbt = torch.zeros(size=(len(stem_to_index), P_wbt.shape[1], P_wbt.shape[2]))

    for stem, i in stem_to_index.items():
        #ptbs[:, :, i] = ptbw[:, :, stem_to_word_indices[stem]].sum(2)
        P_sbt[i, :, :] = P_wbt[stem_to_word_indices[stem], :, :].sum(0)

    # this isn't normalized
    tb = torch.from_numpy(numpy.transpose(cube["window_topic"], [1, 0]))

    # so, normalize it
    P_tb = tb / tb.sum(0).T # P(t|b)
    P_bt = tb.T / tb.sum(1) # P(b|t)

    topic_words = []
    topic_glosses = []
    for tid in range(P_wbt.shape[2]):
        aP_wt = P_sbt[:, :, tid].sum(1) / P_sbt.shape[1]
        top_indices = torch.argsort(aP_wt, dim=0, descending=True)#[0:args.num_top_words]

        topic_table["ids"].append(tid)
        topic_table["top_words"].append([(w.item(), stems[w]) for w in top_indices])
        topic_table["metrics"]["Entropy"].append(torch.distributions.Categorical(aP_wt).entropy().item())
        topic_table["window_weights"].append([max(x, 0.0) for x in P_bt[:, tid].tolist()])
    sP_wbt = torch.argsort(P_wbt, 0, descending=True)
    
    topic_table["window_top_words"] = torch.permute(sP_wbt, [2, 1, 0]).tolist() #.tolist()
    topic_table["window_word_weights"] = P_wbt.tolist()
    for ann in annotations:
        if "topic_id" in ann:            
            topic_table["annotations"][ann["annotator"]] = topic_table["annotations"].get(ann["annotator"], {})
            topic_table["annotations"][ann["annotator"]][ann["topic_id"]] = ann["label"]

        #print(topic)
        #for bid in range(pwbt.shape[1]):
        #    topic[cube["index_to_window"][bid]] = ptb[tid][bid].item()

        #topic_table.append(topic_w)
        #topic_glosses.append(topic_g)
    #topic_table["words"] = topic_words
    return topic_table, word_table, document_table, window_table

    
if __name__ == "__main__":

    import argparse
    import pickle
    import json
    import os.path
    import gzip
    from importlib.resources import files
    import torch
    import numpy
    from jinja2 import BaseLoader, TemplateNotFound
    from flask import Flask, render_template, request

    def meta_open(fname, mode="r"):
        return (gzip.open if fname.endswith("gz") else open)(fname, mode)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", dest="model_file")
    parser.add_argument("--documents_file", dest="documents_file")
    parser.add_argument("--embeddings_file", dest="embeddings_file")
    parser.add_argument("--cube_file", dest="cube_file")
    parser.add_argument("--annotations_file", dest="annotations_file")
    parser.add_argument("--glosses_file", dest="glosses_file")    
    parser.add_argument("--device", dest="device", default="cpu")
    parser.add_argument("--template_overrides_path", dest="template_overrides_path")
    parser.add_argument("--limit_documents", dest="limit_documents", type=int, help="Only load this many documents")
    parser.add_argument("--text_field", dest="text_field", default="content", help="Document field containing the full text (a list of lists of strings)")
    parser.add_argument("--time_field", dest="time_field", default="date", help="Document field containing the numeric time (typically the year)")
    parser.add_argument("--title_field", dest="title_field", default="title", help="Document field containing the title")
    parser.add_argument("--author_field", dest="author_field", default="author", help="Document field containing the author")
    parser.add_argument("--metadata_fields", dest="metadata_fields", nargs="*", default=[], help="Additional document metadata fields to consider")
    args = parser.parse_args()

    trained_model = None
    if args.model_file:
        with meta_open(args.model_file, "rb") as ifd:
            trained_model = torch.load(ifd, map_location=torch.device(args.device), weights_only=False)
        trained_model = trained_model.to(args.device)
        trained_model.eval()
        word_to_index = {w : i for i, w in enumerate(trained_model.word_list)}

    cube = None
    if args.cube_file:
        with meta_open(args.cube_file, "rb") as ifd:
            cube = pickle.load(ifd)

    all_documents = []
    if args.documents_file:
        with meta_open(args.documents_file, "rb") as ifd:
            for i, line in enumerate(ifd):
                if args.limit_documents and i >= args.limit_documents:
                    break
                all_documents.append(json.loads(line))

    embeddings = None
    if args.embeddings_file:
        with meta_open(args.embeddings_file, "rb") as ifd:
            embeddings = pickle.load(ifd)            

    annotations = None
    if args.annotations_file:
        with meta_open(args.annotations_file, "rb") as ifd:
            annotations = [json.loads(l) for l in ifd]

    glosses = None
    if args.glosses_file:
        glosses = {}
        with meta_open(args.glosses_file, "rb") as ifd:
            for line in ifd:
                toks = line.strip().split("\t")
                if len(toks) > 2:
                    word, word2, gloss = toks[:3]
                    word = normalize(word)
                    word2 = normalize(word2)
                    glosses[word.rstrip("-")] = gloss
                    glosses[word2.rstrip("-")] = gloss

    topic_table, word_table, document_table, window_table = preprocess(cube, annotations)
    
    class TemplateLoader(BaseLoader):
        def get_source(self, environment, template):
            if args.template_overrides_path and os.path.exists(os.path.join(args.template_overrides_path, template)):
                override = os.path.join(args.template_overrides_path, template)
                with open(override, "rt") as ifd:
                    source = ifd.read()
            else:
                try:
                    source = files("detm").joinpath("templates/{}".format(template)).read_text()
                except:
                    raise TemplateNotFound(template)
            return source, template, lambda: False

    class App(Flask):
        @property
        def jinja_loader(self):
            return TemplateLoader()
    
    app = App("detm")

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            model=trained_model,
            cube=cube,
            documents=all_documents,
            embeddings=embeddings,
            annotations=annotations,
            glosses=glosses
        )

    @app.route("/model", methods=["GET", "POST"])
    def model():
        if request.method == "POST":
            data_batch = torch.zeros((1, trained_model.vocab_size))
            times_batch = torch.zeros((1, ))
            window = int(request.form.get("window"))
            times_batch[0] = trained_model.min_time + window * trained_model.window_size
            for token in request.form.get("text").lower().split():
                if token in word_to_index:
                    data_batch[0, word_to_index[token]] += 1
            (_, _, _, _, _, topic_mixture_priors, document_topic_mixtures, topic_distributions) = trained_model(data_batch, times_batch)
            print(document_topic_mixtures)
            rel = topic_distributions[window]
            
            return render_template("model.html", model=trained_model, embeddings=embeddings)
        else:
            return render_template("model.html", model=trained_model, embeddings=embeddings)

    @app.route("/topics", methods=["GET", "POST"])
    @app.route("/topics/<int:topic_id>", methods=["GET", "POST"])
    def topics(topic_id=None):
        num_top_words = int(request.args.get("num_top_words", 5))
        if topic_id == None:
            return render_template("topics.html", topics=topic_table, num_top_words=num_top_words)
        else:
            return render_template("topic.html", topics=topic_table, topic_id=topic_id, num_top_words=num_top_words, cube=cube)
        
    @app.route("/words", methods=["GET", "POST"])
    @app.route("/words/<int:word_id>", methods=["GET", "POST"])
    def words(word_id=None):
        if word_id == None:
            sort_by = request.args.get("sort_by", "count")
            per_page = int(request.args.get("per_page", 10))
            page = int(request.args.get("page", 0))
            return render_template("words.html", words=word_table)
        else:
            return render_template("word.html", words=word_table, word_id=word_id)
    
    @app.route("/windows", methods=["GET", "POST"])
    @app.route("/windows/<int:window_id>", methods=["GET", "POST"])
    def windows(window_id=None):
        if window_id == None:
            return render_template("windows.html", cube=cube)
        else:
            return render_template("window.html", cube=cube)
            
    @app.route("/documents", methods=["GET", "POST"])
    @app.route("/documents/<int:doc_id>", methods=["GET", "POST"])
    def documents(doc_id=None):
        if doc_id == None:
            sort_by = request.args.get("sort_by", "count")
            per_page = int(request.args.get("per_page", 10))
            page = int(request.args.get("page", 0))
            return render_template("documents.html", documents=all_documents)
        else:
            doc = all_documents[doc_id]
            title = doc["title"]
            text = " ".join(sum(doc["fullText"], []))
            return render_template("document.html", title=title, text=text)
    
    app.run(port=8080, debug=True)
