import numpy as np

def separate_target(data):
    d = []
    t = []
    for i in data:
        d.append(i[0])
        t.append(i[1])
    return d, t


def batchify(data, target, batch_size):

    data_batch = []
    target_batch = []
    n = int(len(data)/batch_size)

    for i in range(n):
        data_batch.append(data[i*batch_size:(i+1)*batch_size])
        target_batch.append(target[i*batch_size:(i+1)*batch_size])

    return data_batch, target_batch


def vec_data(data, lut, max_l, target):
    new_vec = []
    for line in data[:-1]:
        vec = []
        for word in line.split(' '):
            vec.append(lut[word])
        while len(vec) < max_l:
            vec.append(np.zeros(300))  # todo: set to embedding length
        new_vec.append([vec, target])

    return new_vec
