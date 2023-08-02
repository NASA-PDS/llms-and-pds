import statistics

import numpy as np
import pandas as pd
from numpy.linalg import norm
from pds.llm.tokenization.tokenize import word_tokenize_pds4_xml_files

from wikipedia2vec import Wikipedia2Vec


MODEL_FILE = "./enwiki_20180420_100d.pkl"

wiki2vec = Wikipedia2Vec.load(MODEL_FILE)

URLS = {
    "cassini": "https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml",
    "insight": "https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml",
}

SEARCH_TERMS = [
    "soccer",
    "saturn",
    "cassini",
    "casinni",
    "huygens",
    "orbiter",
    "rss",
    "ionospheric",
    "ionosphere",
    "electron density",
    "insight",
    "context camera",
    "camera",
    "mars",
    "image",
]


def get_label_embeddings(url):
    tokens = word_tokenize_pds4_xml_files(url)
    # Convert tokens to vectors
    vectors = []
    for token in tokens:
        try:
            vectors.append(wiki2vec.get_word_vector(token))
        except KeyError:
            pass
        # not sure if `token in wiki2vec.dictionary` works as we want it to
        # vectors = [wiki2vec.get_word_vector(token) for token in tokens if token in wiki2vec.dictionary]

    return vectors


def get_search_embeddings(term):
    """
    for each token of the search terms, e.g. `context camera` return an embeddings.
    the embedding are stored in a dictionary which key is the search token
    """
    search_embeddings = {}
    for token in term.split(" "):
        try:
            search_embeddings[token] = wiki2vec.get_word_vector(token)
        except KeyError:
            print(f"term {token} not found in model")
            pass
    return search_embeddings


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def get_token_max_similarity(v_ref, vs):
    """
    get the max cosine simlarity between vector v_ref and vectors vs
    """
    cos_sims = [cosine_similarity(v_ref, v) for v in vs]
    return max(cos_sims)


def main():
    label_embbedings = {}
    for k, v in URLS.items():
        label_embbedings[k] = get_label_embeddings(v)

    search_embeddings = {}
    for search_term in SEARCH_TERMS:
        search_embeddings[search_term] = get_search_embeddings(search_term)

    mean_max_sims = []
    for search_term in SEARCH_TERMS:
        line = dict(search_term=search_term)
        for label, url in URLS.items():
            max_similarities = []
            for token, vector in search_embeddings[search_term].items():
                max_similarity = get_token_max_similarity(vector, label_embbedings[label])
                max_similarities.append(max_similarity)
            line[label] = statistics.mean(max_similarities) if max_similarities else np.nan
        mean_max_sims.append(line)

    df = pd.DataFrame(mean_max_sims)
    print(df)
