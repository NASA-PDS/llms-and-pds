import tensorflow_hub as hub
import numpy as np
from pds.llm.tokenization.tokenize import sentence_tokenize_from_pds4_label_url
from numpy.linalg import norm

URLS = [
    'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml',
    'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml'
]


def get_embeddings(url, use_model):
    tokens = sentence_tokenize_from_pds4_label_url(url)
    sentences = [' '.join(tokens)]
    embeddings = use_model(sentences)
    return embeddings


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def cosine_similarity_of_terms(embedding_vectors, use_model):
    search_terms = [
        'soccer',
        'saturn',
        'cassini',
        'cassini',
        'huygens',
        'orbiter',
        'rss',
        'ionospheric',
        'ionosphere',
        'electron density',
        'insight',
        'context camera',
        'camera',
        'mars',
        'image'
    ]
    search_embeddings = {}
    max_cos_sim = {}
    for term in search_terms:
        search_embeddings[term] = use_model([term])
        max_cos_sim[term] = {}
        for url, emb_set in embedding_vectors.items():
            cos_sim = []
            for emb in emb_set:
                cos_sim.append(cosine_similarity(search_embeddings[term][0:], emb))
            max_cos_sim[term][url] = max(cos_sim) if len(cos_sim) > 0 else np.nan
        print(f'{term} {[v for v in max_cos_sim[term].values()]}')

    threshold = 0.6
    match = {}
    for st, url_dict in max_cos_sim.items():
        match[st] = {}
        for url, sim in url_dict.items():
            match[st][url] = sim > threshold
            print(st, url, sim > threshold)
    return max_cos_sim


def main():
    use_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')  # Load the USE model
    embeddings = {}
    for url in URLS:
        embeddings[url] = get_embeddings(url, use_model)
    similarities = cosine_similarity_of_terms(embeddings, use_model)


if __name__ == '__main__':
    main()
