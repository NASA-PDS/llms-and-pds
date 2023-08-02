import logging
import pickle

import requests
import tokenization.pds_tokenizer as pt

from wikipedia2vec import Wikipedia2Vec

logger = logging.getLogger(__name__)


MODEL_FILE = "./enwiki_20180420_100d.pkl"

wiki2vec = Wikipedia2Vec.load(MODEL_FILE)

collection_url = "http://pds.nasa.gov/api/search/1/classes/collections"


def get_wiki2vec_embeddings(tokens: list[str]):
    for token in tokens:
        try:
            yield wiki2vec.get_word_vector(token)
        except KeyError:
            pass


def get_one_page_of_url_embeddings(
    start=1, limit=100, tokenizer=pt.simple_word_tokenizer, get_embeddings=get_wiki2vec_embeddings
):
    params = dict(start=start, limit=limit)
    api_response = requests.get(collection_url, params=params)

    products = api_response.json()["data"]
    urls = [p["properties"]["ops:Label_File_Info.ops:file_ref"][0] for p in products]

    url_embeddings = {}
    for url in urls:
        try:
            url_tokens = tokenizer(url)
            url_embeddings[url] = [e for e in get_embeddings(url_tokens)]
        except (pt.UnReachableCollectionException, pt.UnparsableCollectionException) as e:
            print(str(e))

    return url_embeddings


def main():
    start = 0
    limit = 100
    all_url_embeddings = {}
    while url_embeddings := get_one_page_of_url_embeddings(start=start, limit=limit):
        print(start)
        all_url_embeddings.update(url_embeddings)
        start += limit

    with open("collections_wiki2vec.pkl", "wb") as f:
        pickle.dump(all_url_embeddings, f)


if __name__ == "__main__":
    main()
