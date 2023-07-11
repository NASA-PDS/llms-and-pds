import requests
import xmltodict
import nltk
from gensim.models import Fasttext
import Fasttext

def get_PDS_data (url1, url2):
    url1 = 'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml'
    url2 = 'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml'
    response
def load_fasttext_model(model_path):
    model_path = '/Users/arobinson/Documents/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'
    fasttext.load_model(model_path)
    return model_path






#from gensim.models.fasttext import load_facebook_vectors

# #Load the FastText model
# model_path = '/Users/arobinson/Documents/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'
# fasttext_model = load_facebook_vectors(model_path)

# token = 'europa clipper'
# if token in fasttext_model.wv:
#     embedding = fasttext_model.wv[token]
#     print(token, embedding)




















# import requests
# import xmltodict
# import nltk
# from gensim.models import FastText
# import json
# import numpy as np
#
# url1 = 'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml'
# url2 = 'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml'
#
# response = requests.get(url1, url2)
# xml_data = response.text
# xml_dict = xmltodict.parse(xml_data)
# print(json.dumps(xml_dict, indent=4))
#
# #Convert XML dict to string
# xml_string = json.dumps(xml_dict)
