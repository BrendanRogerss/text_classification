import pickle
import data.build_data as data
import gensim
import logging

# Logging code taken from http://rare-technologies.com/word2vec-tutorial/
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

# Does the model include stop words?
print("Does it include the stop words like \'a\', \'and\', \'the\'? %d %d %d" % (
'a' in model.vocab, 'and' in model.vocab, 'the' in model.vocab))

vocab = data.get_vocab()
lut = {}
for word in vocab:
    lut[word] = model.word_vec(word)

with open('embeddings.pickle', 'wb') as handle:
    pickle.dump(lut, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open("vocabulary.txt", 'wb') as f:
#     # For each word in the current chunk...
#     # Write it out and escape any unicode characters.
#     for i in range(len(vocab)):
#         f.write(vocab[i].encode('UTF-8') + b'\n')
#

