import pickle
import data.build_data as data
import gensim
import numpy as np

print("Loading model")
model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
print("model loaded")
print("building vocab")
same, dif = data.get_vocab()
print("vocab built")
lut = {}
for word in same:
    lut[word] = model.word_vec(word)

for word in dif:
    lut[word] = np.random.uniform(-0.25, 0.25, 300)

with open('lut.p', 'wb') as handle:
    pickle.dump(lut, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open("vocabulary.txt", 'wb') as f:
#     # For each word in the current chunk...
#     # Write it out and escape any unicode characters.
#     for i in range(len(vocab)):
#         f.write(vocab[i].encode('UTF-8') + b'\n')
#
#
