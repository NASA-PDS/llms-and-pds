from transformers import GPT2Tokenizer, GPT2LMHeadModel, FeatureExtractionPipeline
from tokenization.pds_tokenizer import word_tokenize_pds4_xml_files
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

URLS = {
    "cassini": 'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml',
    "insight": 'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml'
}

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
feature_extraction_pipeline = FeatureExtractionPipeline(model=model, tokenizer=tokenizer)

search_terms = [
    "saturn",
    "saturn's rings",
    "cassini",
    "huygens",
    "orbiter",
    "rss",
    "ionospheric",
    "ionosphere",
    "electron density",
    "insight",
    "context camera",
    "camera",
    "mars",
    "image"
]


def clean_tokens_before_embedding(tokens):
    pattern = r"[^a-zA-Z0-9' ]"
    clean_tokens = [re.sub(pattern, '', token) for token in tokens]
    clean_tokens = [token.replace("'", "").replace('"', '') for token in clean_tokens]
    clean_tokens = [token for token in clean_tokens if token.strip()]
    return clean_tokens

def pad_sequences(sequences, max_length, padding_token):
    padded_sequences = [seq[:max_length] + [padding_token] * max(0, max_length - len(seq)) for seq in sequences]
    return padded_sequences


def get_tokens(url):
    tokens = word_tokenize_pds4_xml_files(url)
    clean_tokens = clean_tokens_before_embedding(tokens)
    return clean_tokens


def get_embeddings_for_pds_labels(tokens):
    embeddings = feature_extraction_pipeline(tokens)
    return embeddings


def get_embeddings_for_search_terms(search_terms):
    search_embeddings = feature_extraction_pipeline(search_terms)
    embeddings_list = [item[0] for item in search_embeddings]
    return np.array(embeddings_list)

def get_token_max_similarity(v_ref, vs):
    cos_sims = [cosine_similarity(v_ref, v) for v in vs]
    return max(cos_sims)

def main():
    label_embeddings = {}
    search_term_embeddings = {}
    max_sequence_length = 0

    for label, url in URLS.items():
        tokens = get_tokens(url)
        embeddings = get_embeddings_for_pds_labels(tokens)
        label_embeddings[label] = embeddings
        max_sequence_length = max(max_sequence_length, len(embeddings[0]))

    for label in label_embeddings:
        embeddings = label_embeddings[label]
        padded_embeddings = [e[0] + [0] * (max_sequence_length - len(e[0])) for e in embeddings]
        label_embeddings[label] = np.array(padded_embeddings)

    search_embeddings = get_embeddings_for_search_terms(search_terms)
    search_embeddings = [e[0] + [0] * (max_sequence_length - len(e[0])) for e in search_embeddings]

    mean_max_sims = []
    for i, search_embedding in enumerate(search_embeddings):
        search_term = search_terms[i]
        line = dict(search_term=search_term)
        for label, label_embedding in label_embeddings.items():
            max_similarity = get_token_max_similarity(search_embedding, label_embedding)
            line[label] = max_similarity
        mean_max_sims.append(line)

    df = pd.DataFrame(mean_max_sims)
    print(df)

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
