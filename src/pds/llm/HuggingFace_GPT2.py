"""Runs the cosine similarity comparison test bed using the GPT2 model from HuggingFace."""
from statistics import mean
import numpy as np
import pandas as pd
from numpy.linalg import norm
from tokenization.pds_tokenizer import word_tokenize_pds4_xml_files
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def get_label_embeddings(url):
    """For one pds4 label's URL returns the embeddings of each of its tokens."""
    vectors = []
    tokens = word_tokenize_pds4_xml_files(url)
    for token in tokens:
        text_index = tokenizer.encode(token, add_prefix_space=True)
        vector = model.transformer.wte.weight[text_index, :]
        vectors.append(vector.tolist()[0])
    return vectors


def get_search_embeddings(term):
    """For each token of the search terms, e.g. `context camera` return an embeddings.
    the embedding are stored in a list.

    Args:
        -term- search term for which to compute the embeddings

    Returns:
         The embedding for each word token of the term.

    """
    search_embeddings = []
    for token in term.split(" "):
        text_index = tokenizer.encode(token, add_prefix_space=True)
        vector = model.transformer.wte.weight[text_index, :]
        search_embeddings.append(vector.tolist()[0])
    return search_embeddings


def cosine_similarity(a, b):
    """Returns the cosine similarity between a and b vector (as list).
    a and b must have the same length.

    """
    return np.dot(a, b) / (norm(a) * norm(b))


def get_token_max_similarity(v_ref, vs):
    """Returns the max cosine similarity between vector v_ref and vectors vs."""
    cos_sims = [cosine_similarity(v_ref, v) for v in vs]
    return max(cos_sims)


def get_cosine_similarities(urls, search_terms):
    """Returns panda dataframe of the max averaged cosine similarities between urls and search_terms.
    each url is a column, each search term is a line.
    Values are the max averaged similarities for each PDS4 label url's token (max)
    and each search term, e.g. electron density (average).

    """
    label_embbedings = {}
    for k, v in urls.items():
        label_embbedings[k] = get_label_embeddings(v)

    search_embeddings = {}
    for search_term in search_terms:
        search_embeddings[search_term] = get_search_embeddings(search_term)

    mean_max_sims = []
    for search_term in search_terms:
        line = dict(search_term=search_term)
        for label, _ in urls.items():
            max_similarities = []
            for vector in search_embeddings[search_term]:
                max_similarity = get_token_max_similarity(vector, label_embbedings[label])
                max_similarities.append(max_similarity)
            line[label] = mean(max_similarities) if max_similarities else np.nan
        mean_max_sims.append(line)

    return pd.DataFrame(mean_max_sims)


if __name__ == "__main__":
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
    similarities_df = get_cosine_similarities(URLS, SEARCH_TERMS)
    print(similarities_df)