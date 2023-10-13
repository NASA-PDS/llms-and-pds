# My Project

Code used to experiment with LLM for PDS

## Prerequisites

python 3.9


## Code of Conduct

All users and developers of the NASA-PDS software are expected to abide by our [Code of Conduct](https://github.com/NASA-PDS/.github/blob/main/CODE_OF_CONDUCT.md). Please read this to ensure you understand the expectations of our community.


## Development

To develop this project, use your favorite text editor, or an integrated development environment with Python support, such as [PyCharm](https://www.jetbrains.com/pycharm/).




### Contributing

For information on how to contribute to NASA-PDS codebases please take a look at our [Contributing guidelines](https://github.com/NASA-PDS/.github/blob/main/CONTRIBUTING.md).


### Installation

Install in editable mode and with extra developer dependencies into your virtual environment of choice:

    git clone ...
    python3 -m venv venv
    source venv/bin/activate
    pip install -e '.[dev]'


### Run:

#### Test bed for some LLM models

##### GPT2

    python src/pds/llm/GPT2.py

##### S-BERT

    python src/pds/llm/sbert_test.py



##### Glove model (Work in Progress):

Download model from https://www.kaggle.com/datasets/sadikaljarif/global-vectorglove?resource=download
Save it to `src/pds/llm/models/`:

    python src/pds/llm/Glove.py

##### Wikipedia2vec

Download model from https://wikipedia2vec.github.io/wikipedia2vec/pretrained/ 
Save it do `src/pds/llm/models/`:

    python src/pds/llm/Wiki2Vec.py

##### Others

Other tests have been done with Universal Sentence Encoder model and Fast_Text models.



#### Dynamic search

##### Pre-compute embeddings for collections

    python src/pds/llm/embedding_all_collections.py

##### Search for collections

    python src/pds/llm/find_top_collections_matching.py


    

