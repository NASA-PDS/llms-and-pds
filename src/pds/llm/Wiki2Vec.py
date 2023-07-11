from wikipedia2vec import Wikipedia2Vec
from pds.llm.tokenization.tokenize import tokenize_pds4_xml_files_AR
import numpy as np
from numpy.linalg import norm


MODEL_FILE = '/Users/arobinson/Documents/enwiki_20180420_100d.pkl'
wiki2vec = Wikipedia2Vec.load(MODEL_FILE)
urls = [ 'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml'
            'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml']
def get_embeddings(urls, MODEL_FILE, clean_tokens):
    embedding_vectors = []
    for url in urls:
        tokens = tokenize_pds4_xml_files_AR(url, clean_tokens=[])
        #Convert tokens to vectors
        vectors = [wiki2vec.get_word_vector(tokens)for token in tokens if token in wiki2vec.vocab]
        embedding_vectors.append(vectors)
        print(embedding_vectors)
        return(clean_tokens, embedding_vectors)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def cosine_similarity_of_terms(clean_tokens, embedding_vectors):
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
    for term in search_terms:
        term_vector = wiki2vec.get_word_vector(term)
        min_cos_sim = np.inf
        for emb_set in embedding_vectors:
            for emb in emb_set:
                cos_sim = cosine_similarity(term_vector, emb)
                min_cos_sim = min(min_cos_sim, cos_sim)
        similarities.append((term, min_cos_sim))
    return similarities


def main():
    clean_tokens, embedding_vectors = get_embeddings(urls, MODEL_FILE)
    similarities = cosine_similarity_of_terms(clean_tokens, embedding_vectors)
    for term, similarity in similarities:
        print(f"Cosine similarity for '{term}: {similarity}")

if __name__ == '__main__':
    main()