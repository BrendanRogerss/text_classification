import pickle
import data.build_data
import random


batch_size = 64

def separate_target(data):
    d = []
    t = []
    for i in data:
        d.append(i[0])
        t.append(i[1])
    return d, t


lut = pickle.load(open("lut.p", "rb"))

pos, neg = data.build_data.get_data()


pos_vec = []
for line in pos[:-1]:
    vec = []
    for word in line.split(' '):
        vec.append(lut[word])
    pos_vec.append([vec, [1, 0]])


neg_vec = []
for line in neg[:-1]:
    vec = []
    for word in line.split(' '):
        vec.append(lut[word])
    neg_vec.append([vec, [0, 1]])

data = neg_vec+pos_vec
random.seed(111)
random.shuffle(data)

percent_split = int(len(data)*0.05)
train_data, train_target = separate_target(data[:len(data)-(percent_split*2)])
valid_data, valid_target = separate_target(data[len(data)-(percent_split*2):len(data)-percent_split])
test_data, test_target = separate_target(data[len(data)-percent_split:])


