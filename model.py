import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 100, (3, 300))
        self.conv2 = nn.Conv2d(1, 100, (4, 300))
        self.conv3 = nn.Conv2d(1, 100, (5, 300))

        self.fc1 = nn.Linear(300, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):

        c1 = F.relu(self.conv1(x)).squeeze(3)
        c2 = F.relu(self.conv2(x)).squeeze(3)
        c3 = F.relu(self.conv3(x)).squeeze(3)

        p = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in [c1,c2,c3]]

        x = torch.cat(p,1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x
