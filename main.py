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

pos_vec = util.vec_data(pos, lut, max_l, 0)
neg_vec = util.vec_data(neg, lut, max_l, 1)


data = neg_vec+pos_vec
random.seed(111)
random.shuffle(data)

percent_split = int(len(data)*0.1)
train_data, train_target = util.separate_target(data[:-percent_split])
valid_data, valid_target = util.separate_target(data[-percent_split:])

train_data, train_target = util.batchify(train_data, train_target, batch_size)
valid_data, valid_target = util.batchify(valid_data, valid_target, batch_size)

train_data = torch.Tensor(train_data).cuda()
train_target = torch.Tensor(train_target).long().cuda()

valid_data = torch.Tensor(valid_data).cuda()
valid_target = torch.Tensor(valid_target).long().cuda()

train_data = train_data[:,:,None,:]
valid_data = valid_data[:,:,None,:]

cnn = model.Net()
cnn.cuda()

#output = cnn(train_data[0])
print("Batches: ", train_data.size()[0])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

params = list(cnn.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Model total parameters:', total_params)

def evaluate(net: model.Net):
    net.eval()
    corrects, avg_loss = 0, 0
    for batch in range(valid_data.size()[0]):
        with torch.no_grad():
            inputs = valid_data[batch]
            targets = valid_target[batch]

            logit = net(inputs)
            loss = nn.CrossEntropyLoss()(logit, targets, size_average=False)

            avg_loss += loss
            #corrects += (torch.max(logit, 1)[1].view(targets.size()) == target.data).sum()

    size = len(valid_data.size()[0]*valid_data[1])
    avg_loss /= size
    #accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       0,
                                                                       corrects,
                                                                       0))
    return avg_loss



epochs = 100
print("Started training")
for e in range(epochs):

    running_loss = 0.0

    for batch in range(train_data.size()[0]):
        optimizer.zero_grad()

        inputs = train_data[batch]
        targets = train_target[batch]
        #print(targets.size(), targets)
        outputs = cnn(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch == 190 :
            print('[%d, %5d] loss: %.3f' %
                  (e + 1, batch + 1, running_loss / 190))
            running_loss = 0.0
    #evaluate(cnn)


