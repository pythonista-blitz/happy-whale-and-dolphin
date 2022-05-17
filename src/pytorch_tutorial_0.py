# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.compose import TransformedTargetRegressor

# %%
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor()
    ]
)

train = torchvision.datasets.MNIST(
    root="../dataset/",
    train=True,
    download=True,
    transform=transform
)

# %%
print(len(train))
train[0][0].shape

# %%

# %%

# (0:channels, 1:height, 2:width) -> (1:height, 2:width、0:channels)
img = np.transpose(train[0][0], (1, 2, 0))
img.shape
# %%
img = img.reshape(img.shape[0], img.shape[1])
img.shape
# %%
plt.imshow(img, cmap="gray")
# %%
x = train[0][0]
# %%

# %%
# convolution
conv = nn.Conv2d(
    in_channels=1,
    out_channels=4,
    kernel_size=3,
    stride=1,
    padding=1
)

# %%
conv.weight
# %%
conv.weight.shape
# %%
conv.bias
conv.bias.shape

# %%
x = x.reshape(1, 1, 28, 28)
# %%
x = conv(x)
x
# %%
x.shape
# %%
# pooling
x = F.max_pool2d(x, kernel_size=2, stride=2)
x.shape
# %%
# tensor->vector
x_shape = x.shape[1] * x.shape[2] * x.shape[3]
x_shape
# %%
# 今回はベクトルの要素数が決まっているため、サンプル数は自動で設定
# -1 とするともう片方の要素に合わせて自動的に設定されます
x = x.view(-1, x_shape)
x.shape

# %%
fc = nn.Linear(x_shape, 10)
x = fc(x)
# %%
x.shape
# %%
