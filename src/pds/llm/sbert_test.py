from sentence_transformers import SentenceTransformer
from pds.llm.tokenization.tokenize import tokenize_from_pds4_label_url
import numpy as np
from numpy.linalg import norm

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_label_embeddings(url, max_words=-1):
    sentences = tokenize_from_pds4_label_url(url, max_words=max_words)
    return [model.encode(sentence) for sentence in sentences]


tested_labels = [
    'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml',
    'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml'
]

label_embeddings = {}
for tested_label in tested_labels:
    label_embeddings[tested_label] = get_label_embeddings(tested_label, max_words=-1)


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


search_terms_embeddings = {}
for search_term in search_terms:
    search_terms_embeddings[search_term] = model.encode(search_term)


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


print("search term\turl1\turl2")
for s, se in search_terms_embeddings.items():
    min_cos_sims = []
    for p, les in label_embeddings.items():
        cos_sims = [cosine_similarity(se, le) for le in les]
        min_cos_sims.append(str(min(cos_sims)))
    tab = '\t'
    print(f"{s}{tab}{tab.join(min_cos_sims)}")

print("that's it !")

# products --> n words -->  n embeddings
# search terms --> m words --> m embeddings
# m*n distances (cosine similarities)
# minimum
