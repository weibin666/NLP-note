import torch
from torch import nn
from torch.nn import functional as F
linear=nn.Linear(32,2)
inputs=torch.rand(3,32)
outputs=linear(inputs)
print(outputs)
activation=F.sigmoid(outputs)
print(activation)

