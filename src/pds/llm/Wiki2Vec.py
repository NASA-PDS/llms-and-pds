from wikipedia2vec import Wikipedia2Vec
from pds.llm.tokenization.tokenize import tokenize_pds4_xml_files_AR
import numpy as np
from numpy.linalg import norm


MODEL_FILE = './enwiki_20180420_100d.pkl'
wiki2vec = Wikipedia2Vec.load(MODEL_FILE)
urls = [ 'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml',
            'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml']


def get_embeddings(urls, MODEL_FILE):
    embedding_vectors = {}
    for url in urls:
        tokens = tokenize_pds4_xml_files_AR(url, clean_tokens=[])
        #Convert tokens to vectors
        vectors = []
        for token in tokens:
            try:
                vectors.append(wiki2vec.get_word_vector(token))
            except KeyError:
                pass
        # not sure if `token in wiki2vec.dictionary` works as we want it to
        #vectors = [wiki2vec.get_word_vector(token) for token in tokens if token in wiki2vec.dictionary]
        embedding_vectors[url] = vectors

    return embedding_vectors


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
    similarities = []
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
    return max_cos_sim


def main():
    embedding_vectors = get_embeddings(urls, MODEL_FILE)
    similarities = cosine_similarity_of_terms(embedding_vectors)


if __name__ == '__main__':
    main()