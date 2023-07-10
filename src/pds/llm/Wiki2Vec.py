import requests
import xmltodict
import nltk
from ssl_certificate import certificate
from wikipedia2vec import Wikipedia2Vec
import json

# Set up SSL & download 'punkt'
certificate()

# Saturn Ionosphere Online XML Schema
url1 = 'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml'
url2 = 'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml'

# Retrieve data from XML URL
response = requests.get(url1)
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
clean_tokens = [[token.replace('_', ' ') for token in clean_tokens
                 if '"' not in token and ',' not in token and '@' not in
                 token and '`' not in token and ':' not in token and '{' not in
                 token and "'" not in token and '.' not in token and ';' not in
                 token and '#' not in token and '[' not in token and ']' not in
                 token and '{' not in token and '}' not in token] for clean_tokens in tokens]
print(clean_tokens)

# Load the Wiki2Vec model
MODEL_FILE = '/Users/arobinson/Documents/enwiki_20180420_100d.pkl'
wiki2vec = Wikipedia2Vec.load(MODEL_FILE)


# Embed the tokens using Wiki2Vec
embeddings = []
for sentence_tokens in clean_tokens:
    sentence_embeddings = [wiki2vec.get_word_vector(clean_tokens) for clean_tokens in sentence_tokens if clean_tokens in wiki2vec.dictionary]
    # Use zip to pair up token and vector
    for clean_tokens, embedding in zip(sentence_tokens, sentence_embeddings):
        print(clean_tokens, embedding)
        print()




# word_vector = wiki2vec.get_word_vector('moons')
# similar_words = wiki2vec.most_similar(wiki2vec.get_word('jovian'), 20)
# print("Word vector for 'moons':", word_vector)
# print("Similar words to 'jovian':", similar_words)