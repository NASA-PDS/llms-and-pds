import requests
import xmltodict
import nltk
from ssl_certificate import certificate
from wikipedia2vec import Wikipedia2Vec

# Set up SSL & download 'punkt'
certificate()

# Load the Wiki2Vec model
MODEL_FILE = './enwiki_20180420_100d.pkl'
wiki2vec = Wikipedia2Vec.load(MODEL_FILE)


# Saturn Ionosphere Online XML Schema
url1 = 'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml'
url2 = 'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml'

tested_labels = [url1, url2]








def get_label_embeddings(url):
    # Retrieve data from XML URL
    response = requests.get(url1)
    xml_data = response.text
    xml_dict = xmltodict.parse(xml_data)

    tokens = tokenize_dict(xml_dict, max_words=5)

    return [wiki2vec.get_word_vector(token) for token in tokens if token in wiki2vec.dictionary]

label_embeddings = {}
for tested_label in tested_labels:
    label_embeddings[tested_label] = get_label_embeddings(tested_label)

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
    'elctron density',
    'insight',
    'context camera',
    'camera',
    'mars',
    'image'
]

print("search term \turl1 \t url2")
for search_term in search_terms:
    for tested_label in tested_labels:
        pass


# Embed the tokens using Wiki2Vec
# embeddings = []
# for sentence_tokens in clean_tokens:
#     sentence_embeddings = [wiki2vec.get_word_vector(clean_tokens) for clean_tokens in sentence_tokens if clean_tokens in wiki2vec.dictionary]
#     # Use zip to pair up token and vector
#     for clean_tokens, embedding in zip(sentence_tokens, sentence_embeddings):
#         print(clean_tokens, embedding)
#         print()




# word_vector = wiki2vec.get_word_vector('moons')
# similar_words = wiki2vec.most_similar(wiki2vec.get_word('jovian'), 20)
# print("Word vector for 'moons':", word_vector)
# print("Similar words to 'jovian':", similar_words)
