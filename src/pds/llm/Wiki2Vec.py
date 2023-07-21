from wikipedia2vec import Wikipedia2Vec
from pds.llm.tokenization.pds_tokenizer import word_tokenize_pds4_xml_files
import numpy as np
from numpy.linalg import norm


MODEL_FILE = './enwiki_20180420_100d.pkl'

wiki2vec = Wikipedia2Vec.load(MODEL_FILE)

URLS = ['https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml',
        'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml']


def get_embeddings(url):

    tokens = word_tokenize_pds4_xml_files(url)
    #Convert tokens to vectors
    vectors = []
    for token in tokens:
        try:
            vectors.append(wiki2vec.get_word_vector(token))
        except KeyError:
            pass
        # not sure if `token in wiki2vec.dictionary` works as we want it to
        #vectors = [wiki2vec.get_word_vector(token) for token in tokens if token in wiki2vec.dictionary]

    return vectors


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def cosine_similarity_of_terms(embedding_vectors):
    search_terms = [
        'soccer',
        'saturn',
        'cassini',
        'casinni',
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
        search_embeddings[term] = []
        for token in term.split(" "):
            try:
                search_embeddings[term].append(wiki2vec.get_word_vector(token))
            except KeyError:
                print(f"term {token} not found in model")
                pass
        max_cos_sim[term] = {}
        for url, emb_set in embedding_vectors.items():
            cos_sim = []
            for emb in emb_set:
                for search_emb in search_embeddings[term]:
                   cos_sim.append(cosine_similarity(search_emb, emb))
            max_cos_sim[term][url] = max(cos_sim) if len(cos_sim) > 0 else np.nan

        print(f'{term} {[v for v in max_cos_sim[term].values()]}')

    threshold = 0.6
    match = {}
    for st, url_dict in max_cos_sim.items():
        match[st] = {}
        for url, sim in url_dict.items():
            match[st][url] = sim > threshold
            print(st, url, sim>threshold)
    return max_cos_sim


def main():
    embbedings = {}
    for url in URLS:
        embbedings[url] = get_embeddings(url)

    similarities = cosine_similarity_of_terms(embbedings)


if __name__ == '__main__':
    main()