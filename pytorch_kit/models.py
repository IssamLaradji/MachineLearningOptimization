import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn

import utils as tu
from base import BaseModel
import scipy as sp

#### ---------------- NETWORKS
class AttentionModel(BaseModel):
    def __init__(self, n_channels=1, n_outputs=1):
        super(AttentionModel, self).__init__(problem_type="classification", 
                                             loss_name="binary_crossentropy")

        self.conv1 = nn.Conv2d(n_channels, 20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(40, 1, kernel_size=3, padding=1)
        self.n_outputs = n_outputs


    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.max_pool2d(self.conv3(x), kernel_size=2)
        x = F.sigmoid(x)

        x = F.max_pool2d(x, kernel_size=x.size()[2:])

        return torch.squeeze(x)

    def get_heatmap(self, x, output=1):
        n, _, n_rows, n_cols = x.shape

        x = tu.numpy2var(x)

        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.max_pool2d(self.conv3(x), kernel_size=2)
        x = F.sigmoid(x)

        x = tu.get_numpy(x)

        return x

class LinearModel(BaseModel):
    def __init__(self, n_features=1, n_outputs=10):
        super(LinearModel, self).__init__(problem_type="classification", 
                                        loss_name="categorical_crossentropy")
        self.n_outputs = self.n_classes = n_outputs
        self.fc = nn.Linear(n_features, n_outputs)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x)

class SLFN(BaseModel):
    def __init__(self, n_features=1, n_outputs=10, problem_type="regression"):
        if problem_type == "classification":
            loss_name = "categorical_crossentropy"
        else:
            loss_name = "mse"

        super(SLFN, self).__init__(problem_type=problem_type, 
                                            loss_name=loss_name)
        self.n_outputs = n_outputs
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, n_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        return F.log_softmax(x)

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Siamese(BaseModel):
    def __init__(self):
        super(Siamese, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.MaxPool2d(2, stride=2))

        self.fc1 = nn.Sequential(
            nn.Linear(14450, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            nn.Linear(10, 2))

        # self.cnn2 = self.cnn1
        # self.fc2= self.fc1

        # print self.cnn1 is self.cnn2

    def forward(self, input1, input2):

        output1 = self.cnn1(input1)

        output1 = output1.view(output1.size()[0], -1)
        output1 = self.fc1(output1)

        output2 = self.cnn1(input2)
        output2 = output2.view(output2.size()[0], -1)
        output2 = self.fc1(output2)
        # print output1 - output2
        output = torch.sqrt(torch.sum((output1 - output2) * (output1 - output2), 1))

        return output

class cifar_net(BaseModel):
    def __init__(self, n_channels=3, n_classes=10, problem_type="classification"):
        super(cifar_net, self).__init__(problem_type=problem_type, 
                                            loss_name="categorical_crossentropy")

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):        
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = F.relu(self.pool(self.conv4(x)))

        x = x.view(-1, 4096)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x)


class SmallUNet(BaseModel):
    def __init__(self, n_channels, n_classes,loss_name="dice_loss"):
        sup = super(SmallUNet, self)
        sup.__init__(problem_type="segmentation",
                     loss_name=loss_name)

        self.n_outputs = n_classes
        self.n_channels = n_channels

        k = 5
        self.conv1 = nn.Conv2d(n_channels, k, 3, padding=1)
        self.conv2 = nn.Conv2d(k, k, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = nn.Conv2d(k, k*2, 3, padding=1)
        self.conv4 = nn.Conv2d(k*2, k*2, 3, padding=1)
        self.conv5 = nn.Conv2d(k*2, k, 3, padding=1)
        self.conv6 = nn.Conv2d(k*2, k, 3, padding=1)
        self.conv7 = nn.Conv2d(k, n_classes, 3, padding=1)
        self.dropout2d = nn.Dropout2d(p=0.5)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = self.dropout2d(x)
        x1 = self.pool(x)
        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))
        x1 = F.relu(self.conv5(x1))
        #x1 = self.dropout2d(x1)
        x1 = self.upsample(x1)
        x = torch.cat([x, x1], 1)
        x = F.relu(self.conv6(x))
        #x = self.dropout2d(x)
        x = self.conv7(x)

        return F.sigmoid(x)


class SimpleConvnet(BaseModel):
    # GOOD FOR MNIST
    def __init__(self, n_channels=1, n_classes=10, problem_type="classification"):
        super(SimpleConvnet, self).__init__(problem_type=problem_type, 
                                            loss_name="categorical_crossentropy")

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
