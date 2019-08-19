from data import mydatasets
import torchtext.data as data
import pickle
import data.build_data as data

batch_size = 64


# # load MR dataset
# def mr(text_field, label_field, **kargs):
#     train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
#     text_field.build_vocab(train_data, dev_data)
#     label_field.build_vocab(train_data, dev_data)
#     train_iter, dev_iter = data.Iterator.splits(
#                                 (train_data, dev_data),
#                                 batch_sizes=(batch_size, len(dev_data)),
#                                 **kargs)
#     return train_iter, dev_iter
#
# # load data
# print("\nLoading data...")
# text_field = data.Field(lower=True)
# label_field = data.Field(sequential=False)
# train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
#
# for batch in train_iter:
#     feature, target = batch.text, batch.label
#     feature.data.t_(), target.data.sub_(1)  # batch first, index align
#     print(feature[0], target[0])
#

lut = pickle.load(open("lut.p", "rb"))

data.get_data()