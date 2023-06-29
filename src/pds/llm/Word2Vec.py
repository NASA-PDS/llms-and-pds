import requests
import xmltodict
import nltk
from ssl_certificate import certificate
from gensim.models import Word2Vec
import json
#import re
#numpy. (idea of how close 2 vectors are)
#Set up SSL & download 'punkt'
certificate()

# Saturn Ionosphere Online XML Schema
#url = 'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/xml_schema/collection_schema_corss_saturn_ionosphere.xml'
url = 'https://atmos.nmsu.edu/PDS/data/PDS4/go_ppr_bundle/xml_schema/collection_goppr_xml_schema.xml'

#Retrive data from XML URL
response = requests.get(url)
xml_data = response.text
xml_dict = xmltodict.parse(xml_data)
print(json.dumps(xml_dict, indent=4))

#Convert XML dict to string
xml_string = json.dumps(xml_dict)

#Tokenize XML data
sentences = nltk.sent_tokenize(str(xml_string))
tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
print(tokens)

#Clean the tokens to rid of any non-alphabet/numerical characters
#https://dylancastillo.co/nlp-snippets-clean-and-tokenize-text-with-python/#remove-a-tags-but-keep-its-content
clean_tokens = [[token for token in sentence_tokens if '"' not in token and ',' not in token and '@' not in token and '`' not in token and ':' not in token and '{' not in token and "'" not in token and '.' not in token and ';' not in token and '#' not in token and '[' not in token and ']' not in token and '{' not in token and '}' not in token] for sentence_tokens in tokens]
print(clean_tokens)
#

#Train the Word2Vec model
word2vec_model = Word2Vec(tokens, min_count=1)

#Embed the tokens using Word2Vec
embeddings = []
for sentence_tokens in clean_tokens:
    sentence_embeddings = [word2vec_model.wv[clean_tokens] for clean_tokens in sentence_tokens if clean_tokens in word2vec_model.wv]
    if sentence_embeddings:
        embeddings.append(sentence_embeddings)
        print("Embeddings:", sentence_embeddings)
        #Use zip to pair up token and vector
        for token, embedding in zip(sentence_tokens, sentence_embeddings):
            print(token, embedding)
            print()
#Should I also delete '_' and add spaces for more vectors on those groups of words