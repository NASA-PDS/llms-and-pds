from transformers import GPT2Model, GPT2Tokenizer
import torch
import requests
import xmltodict
import re

def retrieve_data_from_urls(urls):
    data = {}
    for url in urls:
        response = requests.get(url)
        xml_data = response.text
        xml_dict = xmltodict.parse(xml_data)
        text_content = extract_text_content(xml_dict)
        data[url] = text_content
    return data

def extract_text_content(xml_dict):
    if isinstance(xml_dict, str):
        return xml_dict
    elif isinstance(xml_dict, list):
        return ' '.join([extract_text_content(item) for item in xml_dict])
    else:
        return ' '.join([extract_text_content(value) for value in xml_dict.values()])

def tokenize_and_encode_data(data, tokenizer):
    encoded_inputs = tokenizer(list(data.values()), return_tensors='pt', padding=True, truncation=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return encoded_inputs

# PDS4 XML Data URLs
urls = [
    'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml',
    'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml'
]

# Initialize the GPT2 tokenizer & model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Retrieve PDS4 data from URLs
pds4_data = retrieve_data_from_urls(urls)

# Tokenize and embed the PDS4 XML data
encoded_inputs = tokenize_and_encode_data(pds4_data, tokenizer)

# Generate embeddings
with torch.no_grad():  # Disable gradient calculation
    outputs = model(**encoded_inputs)
embeddings = outputs.last_hidden_state

print(embeddings)
