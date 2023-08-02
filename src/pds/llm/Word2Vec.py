import json

import nltk
import numpy as np
import requests
import xmltodict
from gensim.models import Word2Vec
from ssl_certificate import certificate

# Set up SSL & download 'punkt'
certificate()

# Saturn Ionosphere Online XML Schema
url1 = "https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml"
url2 = "https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml"

# Retrive data from XML URL
response = requests.get(url1, url2)
xml_data = response.text
xml_dict = xmltodict.parse(xml_data)
print(json.dumps(xml_dict, indent=4))

# Convert XML dict to string
xml_string = json.dumps(xml_dict)

# Tokenize XML data
sentences = nltk.sent_tokenize(str(xml_string))
tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
print(tokens)

# Clean the tokens to rid of any non-alphabet/numerical characters
# https://dylancastillo.co/nlp-snippets-clean-and-tokenize-text-with-python/#remove-a-tags-but-keep-its-content
clean_tokens = [
    token
    for sentence_tokens in tokens
    for token in sentence_tokens
    if '"' not in token
    and "," not in token
    and "@" not in token
    and "`" not in token
    and ":" not in token
    and "{" not in token
    and "'" not in token
    and "." not in token
    and ";" not in token
    and "#" not in token
    and "[" not in token
    and "]" not in token
    and "{" not in token
    and "}" not in token
    and "*" not in token
]
print(clean_tokens)


# Train the Word2Vec model
word2vec_model = Word2Vec(tokens, min_count=1)

# Embed the tokens using Word2Vec
for token in clean_tokens:
    if token in word2vec_model.wv:
        embedding = word2vec_model.wv[token]
        print(token, embedding)
        print()

# search_terms = ['Cassini']
# distances = []
# for term in search_terms:
#     term_embedding = word2vec_model.wv[term]
#     term_distances = [np.dot(term_embedding, embedding) for embedding in term_embedding]
#     distances.append(term_distances)
#
# print("Distances between embeddings & Search Terms:")
# for term, term_distances in zip(search_terms, distances):
#     print(F"{term}: {term_distances}")
