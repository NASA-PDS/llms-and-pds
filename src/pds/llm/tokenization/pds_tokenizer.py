import requests
import xmltodict
import xml
import json
from nltk.tokenize import word_tokenize, sent_tokenize


def contains_symbol(token):
    symbols = '",:{`.;#[]{*='
    return any([symbol in token for symbol in symbols])


class UnReachableCollectionException(Exception):
    pass


class UnparsableCollectionException(Exception):
    pass


def word_tokenize_pds4_xml_files(url):
    try:
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            xml_data = response.text
            xml_dict = xmltodict.parse(xml_data)
            xml_string = json.dumps(xml_dict)
            sentences = sent_tokenize(xml_string)
            tokens = [word_tokenize(sentence) for sentence in sentences]

            clean_tokens = [token.lower() for sentence_tokens in tokens for token in sentence_tokens if not contains_symbol(token)]
            return clean_tokens
        else:
            raise UnReachableCollectionException(f"Collection {url} can not be reached, status {response.status_code}")
    except requests.exceptions.MissingSchema as e:
        raise UnReachableCollectionException(f"Collection {url} missing http  schema")


def sentence_tokenize_from_pds4_label_url(url, max_words=-1):
    response = requests.get(url)
    xml_data = response.text
    xml_dict = xmltodict.parse(xml_data)

    return tokenize_dict(xml_dict, max_words=max_words)


def simple_word_tokenizer(url):
    try:
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            try:
                xml_data = response.text
                xml_dict = xmltodict.parse(xml_data)
                return word_tokenize_dict(xml_dict)
            except xml.parsers.expat.ExpatError as e:
                raise UnparsableCollectionException(f"Collection {url} is not parsable")
        else:
            raise UnReachableCollectionException(f"Collection {url} is unreachable, status code {response.status_code}")
    except requests.exceptions.MissingSchema as e:
        raise UnReachableCollectionException(f"Collection {url} missing http  schema")


def word_tokenize_dict(d):
    if isinstance(d, str):
        return [t for t in word_tokenize(d) if t not in ':(),.*']
    elif isinstance(d, list):
        tokens = []
        for e in d:
            if e is not None:
                tokens.extend(word_tokenize_dict(e))
        return tokens
    else:
        tokens = []
        for k, v in d.items():
            if v is not None:
                tokens.extend(word_tokenize_dict(v))
        return tokens


def tokenize_dict(d, prefix=[], max_words=-1):
    if isinstance(d, str):
        if max_words != -1:
           words = d.split(" ")
           return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
        else:
            prefix_str = " ".join(prefix[-2:])
            return [f"{prefix_str} {d}".strip()]
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
                prefix.append(f)
                new_tokens = tokenize_dict(v, prefix=prefix, max_words=max_words)
                tokens.extend(new_tokens)
        return tokens
