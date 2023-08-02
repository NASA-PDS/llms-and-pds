import pickle
import statistics

from nltk.tokenize import word_tokenize
from pds.llm.Wiki2Vec import cosine_similarity

from wikipedia2vec import Wikipedia2Vec

MODEL_FILE = "./enwiki_20180420_100d.pkl"
wiki2vec = Wikipedia2Vec.load(MODEL_FILE)


with open("./collections_wiki2vec.pkl", "rb") as f:
    collection_embeddings = pickle.load(f)


def search_for(search_term: str):
    search_tokens = word_tokenize(search_term)
    collection_sims = []
    for url, embeddings in collection_embeddings.items():
        collection_sim = dict(url=url, sims=[])
        for token in search_tokens:
            token_embedding = wiki2vec.get_word_vector(token)
            best_sim = max([cosine_similarity(token_embedding, e) for e in embeddings])
            collection_sim["sims"].append(best_sim)
        collection_sims.append(collection_sim)

    collection_sims.sort(key=lambda x: statistics.mean(x["sims"]), reverse=True)

    return collection_sims


while True:
    search_term = input("what are you searching for ?")
    print(search_for(search_term)[:10])
