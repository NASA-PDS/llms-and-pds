<<<<<<< HEAD
from tokenization.pds_tokenizer import word_tokenize_pds4_xml_files
=======
from pds.llm.tokenization.pds_tokenizer import sentence_tokenize_from_pds4_label_url
>>>>>>> 9bd18d5 (test/fix some model run)
import numpy as np
import os
from numpy.linalg import norm

URLS = ['https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml',
        'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml']

def get_embeddings(url, word_vectors):
    tokens = word_tokenize_pds4_xml_files(url)
    vectors = []
    for token in tokens:
        try:
            vectors.append(word_vectors[token])
        except KeyError:
            pass
    return vectors

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def cosine_similarity_of_terms(embedding_vectors, word_vectors):
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
                search_embeddings[term].append(word_vectors[token])
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
            print(st, url, sim > threshold)
    return max_cos_sim

def main():
<<<<<<< HEAD
    glove_file = '/Users/arobinson/Documents/glove.840B.300d.txt'
=======
    glove_file = os.path.join(
        os.path.dirname(__file__),
        "models/glove.840B.300d.txt"
    )
>>>>>>> 9bd18d5 (test/fix some model run)
    word_vectors = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                word_vectors[word] = coefs
            except ValueError:
                pass

    embeddings = {}
    for url in URLS:
        embeddings[url] = get_embeddings(url, word_vectors)
    similarities = cosine_similarity_of_terms(embeddings, word_vectors)

if __name__ == '__main__':
    main()

'''
Glove files can be downloaded at the following:
https://github.com/stanfordnlp/GloVe
'''