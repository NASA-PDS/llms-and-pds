from wikipedia2vec import Wikipedia2Vec
#Downloaded file from Wikipedia2vec/https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
MODEL_FILE = '/Users/arobinson/Documents/enwiki_20180420_100d.pkl'
wiki2vec = Wikipedia2Vec.load(MODEL_FILE)
word_vector = wiki2vec.get_word_vector('moons')
similar_words = wiki2vec.most_similar(wiki2vec.get_word('jovian'), 20)
print("Word vector for 'moons':", word_vector)
print("Similar words to 'jovian':", similar_words)