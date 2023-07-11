import requests
import xmltodict
import json
from nltk.tokenize import word_tokenize, sent_tokenize

def tokenize_pds4_xml_files_AR(urls):
    response = requests.get(urls)
    xml_data = response.text
    xml_dict = xmltodict.parse(xml_data)
    xml_string = json.dumps(xml_dict)
    sentences = sent_tokenize(xml_string)
    tokens = [word_tokenize(sentence) for sentence in sentences]
    clean_tokens = []
    clean_tokens = [token for sentence_tokens in tokens for token in sentence_tokens if
                    '"' not in token and ',' not in token and '@' not in
                    token and '`' not in token and ':' not in token and '{' not in
                    token and "'" not in token and '.' not in token and ';' not in
                    token and '#' not in token and '[' not in token and ']' not in
                    token and '{' not in token and '}' not in token and '*' not in token and
                    '=' not in token]
    print(clean_tokens)

urls = 'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml'
tokens = tokenize_pds4_xml_files_AR(urls)
print(tokens)