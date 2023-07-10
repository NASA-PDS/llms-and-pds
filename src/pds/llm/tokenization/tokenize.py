import requests
import xmltodict


def tokenize_from_pds4_label_url(url, max_words=-1):
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