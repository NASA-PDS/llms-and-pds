import requests
import xmltodict
import nltk
nltk.download('punkt')
import json
from wikipedia2vec import Wikipedia2Vec

# Saturn Ionosphere Online XML Schema
url = 'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/xml_schema/collection_schema_corss_saturn_ionosphere.xml'

# Retrieve data from XML URL
response = requests.get(url)
xml_data = response.text
xml_dict = xmltodict.parse(xml_data)
print(json.dumps(xml_dict, indent=4))

def tokenize_dict(d, prefix="", max_words=-1):
    if isinstance(d, str):
        #if max_words!=-1:
        #    words = d.split(" ")
        #    token_number = max(len(words) - max_words, 1)
        #    return [" ".join(words[i:min(i+max_words, len(words))]) for i in range(token_number)]
        #else:
        return [f"{prefix} {d}".strip()]
    if isinstance(d, list):
        return [tokenize_dict(e, prefix=prefix, max_words=max_words) for e in d]
    else:
        tokens = []
        for f, v in d.items():
            new_tokens = tokenize_dict(v, prefix=f"{prefix} {f}", max_words=max_words)
            tokens.extend(new_tokens)
        return tokens


tokens = tokenize_dict(xml_dict, max_words=5)

# Tokenize the XML data
#sentences = nltk.sent_tokenize(xml_data)
#tokens = [nltk.word_tokenize(sentence) for sentence in sentences]

# Load the pre-trained embedding model
# downalod it with command
#
model_path = '/Users/arobinson/Documents/enwiki_20180420_100d.pkl'
wikipedia2vec_model = Wikipedia2Vec.load(model_path)

# Embed the tokens using Wikipedia2Vec
embeddings = []
for sentence_tokens in tokens:
    sentence_embeddings = [wikipedia2vec_model.get_word_vector(token) for token in sentence_tokens if token in wikipedia2vec_model]
    if sentence_embeddings:
        embeddings.append(sentence_embeddings)
