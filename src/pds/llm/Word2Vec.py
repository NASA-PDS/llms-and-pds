import requests
import xmltodict
import nltk
from ssl_certificate import certificate
from gensim.models import Word2Vec
import json

#Set up SSL & download 'punkt'
certificate()

# Saturn Ionosphere Online XML Schema
url = 'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/xml_schema/collection_schema_corss_saturn_ionosphere.xml'

#Retrive data from XML URL
response = requests.get(url)
xml_data = response.text
xml_dict = xmltodict.parse(xml_data)
print(json.dumps(xml_dict, indent=4))

#Tokenize XML data
sentences = nltk.sent_tokenize(xml_data)
tokens = [nltk.word_tokenize(sentence) for sentence in sentences]

#Train the Word2Vec model
word2vec_model = Word2Vec(tokens, min_count=1)

#Embed the tokens using Word2Vec
embeddings = []
for sentence_tokens in tokens:
    sentence_embeddings = [word2vec_model.wv[token] for token in sentence_tokens if token in word2vec_model.wv]
    if sentence_embeddings:
        embeddings.append(sentence_embeddings)
        print("Embeddings:", sentence_embeddings)
        #Use zip to pair up token and vector
        for token, embedding in zip(sentence_tokens, sentence_embeddings):
            print(token, embedding)
            print()
#Will need to figure out a way to eliminate '<' and '_' from the XML file before
#Tokenizing the data for embedding & vector purposes