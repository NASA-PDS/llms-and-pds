import torch
from transformers import GPT2Tokenizer, GPT2Model
import requests
import numpy as np
from numpy.linalg import norm
from .tokenization.pds_tokenizer import sentence_tokenize_from_pds4_label_url

URLS = [
    'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml',
    'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml'
]

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load GPT-2 model
model = GPT2Model.from_pretrained('gpt2')

# Function to tokenize and embed PDS4 XML URL
def tokenize_and_embed(url):
    # Get XML content from the URL
    response = requests.get(url)
    xml_content = response.text

    # Tokenize the XML content using your existing function
    tokens = sentence_tokenize_from_pds4_label_url(xml_content)

    # Embed the tokens using GPT-2 model
    input_ids = tokenizer.encode(tokens, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs[0].squeeze(0)  # Remove the batch dimension

    return embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def cosine_similarity_of_terms(embedding_vectors):
    search_terms = [
        'soccer',
        'saturn',
        'cassini',
        'casinni',
        'huygens',
        'orbiter',
        'rss',
        'ionospheric',
        'ionosphere',
        'electron density',
        'insight',
        'context camera',
        'camera',
        'mars',
        'image'
    ]
    search_embeddings = {}
    max_cos_sim = {}
    for term in search_terms:
        search_embeddings[term] = []
        for token in term.split(" "):
            try:
                search_embeddings[term].append(embedding_vectors[token])
            except KeyError:
                print(f"term {token} not found in model")
                pass
        max_cos_sim[term] = {}
        for url, emb_set in embedding_vectors.items():
            cos_sim = []
            for emb in emb_set:
                for search_emb in search_embeddings[term]:
                    cos_sim.append(cosine_similarity(search_emb, emb))
            max_cos_sim[term][url] = max(cos_sim) if len(cos_sim) > 0 else np.nan

        print(f'{term} {[v for v in max_cos_sim[term].values()]}')

    threshold = 0.6
    match = {}
    for st, url_dict in max_cos_sim.items():
        match[st] = {}
        for url, sim in url_dict.items():
            match[st][url] = sim > threshold
            print(st, url, sim > threshold)
    return max_cos_sim

def main():
    embeddings = {}
    for url in URLS:
        embeddings[url] = tokenize_and_embed(url)
    similarities = cosine_similarity_of_terms(embeddings)


if __name__ == '__main__':
    main()