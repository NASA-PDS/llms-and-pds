from transformers import GPT2Tokenizer, FeatureExtractionPipeline, TFGPT2Model
import requests
import json
import xmltodict

URLS = [
    'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml',
    'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml'
]
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
def get_pds4_xml_data(url):
    response = requests.get(url)
    xml_data = response.text
    return xml_data

def tokenize_pds4_xml_files(xml_data):
    xml_dict = xmltodict.parse(xml_data)
    xml_string = json.dumps(xml_dict)
    pds_tokens = tokenizer(xml_string, return_tensors='pt').input_ids
    return pds_tokens

def get_chunks_of_tokens():
    chunks_of_tokens = []
    for url in URLS:
        xml_label_data = get_pds4_xml_data(url)
        xml_tokens = tokenize_pds4_xml_files(xml_label_data)
        max_sequence_length_of_GPT2 = 1024
        chunks_of_xml_labels = [xml_tokens[:, i:i + max_sequence_length_of_GPT2] for i in range(0, xml_tokens.shape[1], max_sequence_length_of_GPT2)]
        chunks_of_tokens.extend(chunks_of_xml_labels)

    return chunks_of_tokens

def find_cosine_similarity_of_pds4_tokens(search_terms, embeddings):
    feature_extraction_pipeline = FeatureExtractionPipeline(model=TFGPT2Model.from_pretrained('gpt2'), tokenizer=tokenizer)

