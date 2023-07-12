import requests
import xmltodict
import json
from nltk.tokenize import word_tokenize, sent_tokenize


def contains_symbol(token):
    symbols = '",:{`.;#[]{*='
    return any([symbol in token for symbol in symbols])


def word_tokenize_pds4_xml_files(url):
    response = requests.get(url)
    xml_data = response.text
    xml_dict = xmltodict.parse(xml_data)
    xml_string = json.dumps(xml_dict)
    sentences = sent_tokenize(xml_string)
    tokens = [word_tokenize(sentence) for sentence in sentences]

    clean_tokens = [token.lower() for sentence_tokens in tokens for token in sentence_tokens if not contains_symbol(token)]
    return clean_tokens


def sentence_tokenize_from_pds4_label_url(url, max_words=-1):
    response = requests.get(url)
    xml_data = response.text
    xml_dict = xmltodict.parse(xml_data)

    return tokenize_dict(xml_dict, max_words=max_words)


def tokenize_dict(d, prefix="", max_words=-1):
    if isinstance(d, str):
        if max_words!=-1:
           words = d.split(" ")
           return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
        else:
            return [f"{prefix} {d}".strip()]
    else:
        if isinstance(d, list):
            tokens = []
            for e in d:
                new_tokens = tokenize_dict(e, prefix=prefix, max_words=max_words)
                tokens.extend(new_tokens)
            return tokens
        else:
            tokens = []
            for f, v in d.items():
                new_tokens = tokenize_dict(v, prefix=f"{prefix} {f}", max_words=max_words)
                tokens.extend(new_tokens)
        return tokens