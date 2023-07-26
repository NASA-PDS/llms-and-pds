from transformers import GPT2Tokenizer, GPT2LMHeadModel, FeatureExtractionPipeline
import requests
import json
import xmltodict


'''
URLS = [
    'https://atmos.nmsu.edu/PDS/data/PDS4/saturn_iono/data/rss_s10_r007_ne_e.xml',
    'https://planetarydata.jpl.nasa.gov/img/data/nsyt/insight_cameras/data/sol/0024/mipl/edr/icc/C000M0024_598662821EDR_F0000_0558M2.xml'
]

def get_pds4_xml_data(url):
    response = requests.get(url)
    xml_data = response.text
    return xml_data

def tokenize_pds4_xml_files(xml_data):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    xml_dict = xmltodict.parse(xml_data)
    xml_string = json.dumps(xml_dict)
    pds_tokens = tokenizer(xml_string, return_tensors='pt').input_ids
    return pds_tokens

for url in URLS:
    xml_label_data = get_pds4_xml_data(url)
    xml_tokens = tokenize_pds4_xml_files(xml_label_data)
    print(f"Number of tokens for '{url}':", len(xml_tokens))
    print("Token IDs:", xml_tokens)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    max_sequence_length_of_GPT2 = 1024
    chunks = [xml_tokens[:, i:i+max_sequence_length_of_GPT2] for i in range(0, xml_tokens.shape[1], max_sequence_length_of_GPT2)]
    for chunk_id, chunk in enumerate(chunks):
        print(f"Number of tokens for '{url}' (chunk {chunk_id}):", len(chunk[0]))
        decoded_tokens = tokenizer.decode(chunk[0], skip_special_tokens=True)
        print("Decoded Text:", decoded_tokens)
'''
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