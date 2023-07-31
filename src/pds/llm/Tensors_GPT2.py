from transformers import GPT2Tokenizer, GPT2Model
import requests
import json
import xmltodict
import numpy as np
from numpy.linalg import norm
import statistics
import pandas as pd
from scipy.sparse import csr_matrix

URLS = {
    "cassini": 'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml',
    "insight": 'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml'
}
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
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

def get_pds4_xml_data(url):
    response = requests.get(url)
    xml_data = response.text
    return xml_data

def parse_xml_like_data(xml_data):
    lines = xml_data.strip().split('\n')
    data_dict = {}
    for line in lines:
        key_value = line.split(' ', 1)
        if len(key_value) == 2:
            key, value = key_value
            try:
                data_dict[key] = csr_matrix([float(value)])
            except ValueError:
                pass
    return data_dict


def get_tensors_for_pds_labels(data_dict):
    xml_dict = xmltodict.parse(data_dict)
    xml_string = json.dumps(xml_dict)
    pds_tokens = tokenizer(xml_string, return_tensors='pt').input_ids
    print("PDS4 Label Tensors:")
    print(pds_tokens)
    return pds_tokens


def get_tensors_for_search_terms():
    search_term_tokens = [tokenizer(term, return_tensors='pt').input_ids for term in search_terms]
    print("Search Term Tensors:")
    for i, term in enumerate(search_terms):
        print(f"Tensor for '{term}':")
        print(search_term_tokens[i])
        # Convert dense tensor to sparse tensor (CSR format)
        search_term_tokens[i] = csr_matrix(search_term_tokens[i].numpy())
    return search_term_tokens


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def get_token_max_similarity(v_ref, vs):
    cos_sims = [cosine_similarity(v_ref, v) for v in vs.values()]
    if not cos_sims:
        return 0.0
    return max(cos_sims)


def main():
    label_embeddings = {}
    for label, url in URLS.items():
        xml_data = get_pds4_xml_data(url)
        data_dict = parse_xml_like_data(xml_data)
        label_embeddings[label] = data_dict

    search_embeddings = get_tensors_for_search_terms()

    mean_max_sims = []
    for term in search_terms:
        line = dict(search_term=term)
        for label, url in URLS.items():
            max_similarities = []
            for vector in search_embeddings:
                max_similarity = get_token_max_similarity(vector, label_embeddings[label])
                max_similarities.append(max_similarity)
            line[label] = statistics.mean(max_similarities) if max_similarities else np.nan
        mean_max_sims.append(line)

    df = pd.DataFrame(mean_max_sims)
    print(df)

if __name__ == '__main__':
    main()

'''
#Prints embeddings using Pipeline
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
feature_extraction_pipeline = FeatureExtractionPipeline(model=model, tokenizer=tokenizer)
word = "Orion"
word_embeddings = feature_extraction_pipeline(word)
embedding_vector = word_embeddings[0]
print("Embedding Vector:", embedding_vector)
#feature_extraction_pipeline(word)[0]
'''

'''Returns vector/tensor of a given word & use numpy to convert to embedding
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text_index = tokenizer.encode('Saturn',add_prefix_space=True)#Requires prefix space for single words
vector = model.transformer.wte.weight[text_index,:]
vector_np = vector.detach().numpy()
print(vector_np)
'''


'''
The following gets tensors for models & text generation for a given topic

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2Model.from_pretrained('gpt2')
text = "I am a intern at NASA JPL."
#Tensorflow tensors represent numerical features/word embeddings
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
print(output)

generator = pipeline('text-generation', model='gpt2')
#Set seed for reproductability
set_seed(42)
generator("The Jet Propulsion Laboratory", max_length=30, num_return_sequences=5)
'''