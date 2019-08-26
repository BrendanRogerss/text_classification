import pickle
import data.build_data
import random
import util
import model
import torch
import torch.optim as optim
import torch.nn as nn

batch_size = 50

lut = pickle.load(open("lut.p", "rb"))


pos, neg = data.build_data.get_data()

max_l = max([len(i.split(' ')) for i in pos+neg])

pos_vec = util.vec_data(pos, lut, max_l, [1, 0])
neg_vec = util.vec_data(neg, lut, max_l, [0, 1])


data = neg_vec+pos_vec
random.seed(111)
random.shuffle(data)

percent_split = int(len(data)*0.1)
train_data, train_target = util.separate_target(data[:-percent_split])
valid_data, valid_target = util.separate_target(data[-percent_split:])
train_data, train_target = util.batchify(train_data, train_target, batch_size)
train_data = torch.Tensor(train_data)
train_target = torch.Tensor(train_target).long()
train_data = train_data[:,:,None,:]

cnn = model.Net()
cnn.cuda()
train_data = train_data.cuda()
train_target = train_target.cuda()
#output = cnn(train_data[0])
#print(output)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

epochs = 10
for e in range(epochs):

    running_loss = 0.0

    for batch in range(train_data.size()[0]):
        optimizer.zero_grad()

        inputs = train_data[batch]
        targets = train_target[batch]
        print(targets.size(), targets)
        outputs = cnn(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch % 100 == 0:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (e + 1, batch + 1, running_loss / 100))
            running_loss = 0.0