# Discriminator class
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes=2):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv1 = nn.Conv2d(1, 100, (3, embed_size))
        self.conv2 = nn.Conv2d(1, 100, (4, embed_size))
        self.conv3 = nn.Conv2d(1, 100, (5, embed_size))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv1)
        x2 = self.conv_and_pool(x, self.conv2)
        x3 = self.conv_and_pool(x, self.conv3)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)
