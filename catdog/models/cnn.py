import torch
import torch.nn as nn


EMBEDDING_SIZE = 128
NUM_CLASSES = 2


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


model_cnn = nn.Sequential()

# Your code here: CONV->POOL->CONV-POOL->... as many as you wish
# End of your code here
model_cnn.add_module("conv_1", nn.Conv2d(3, 6, 5))
model_cnn.add_module("relu_1", nn.ReLU())
model_cnn.add_module("max_pooling_1", nn.MaxPool2d(2, 2))

model_cnn.add_module("conv_2", nn.Conv2d(6, 16, 5))
model_cnn.add_module("relu_2", nn.ReLU())
model_cnn.add_module("max_pooling_2", nn.MaxPool2d(2, 2))

# global max pooling
model_cnn.add_module("global_max_pooling", nn.AdaptiveMaxPool2d(4))
# dropout for regularization
model_cnn.add_module("dropout", nn.Dropout(0.3))
# "flatten" the data
model_cnn.add_module("flat", Flatten())

# last fully-connected layer, used to create embedding vectors
model_cnn.add_module("fc", nn.Linear(256, EMBEDDING_SIZE))
model_cnn.add_module("relu", nn.ReLU())

model_cnn.add_module("dropout_6", nn.Dropout(0.3))

# logits for NUM_CLASSES=2 classes
model_cnn.add_module("fc_logits", nn.Linear(EMBEDDING_SIZE, NUM_CLASSES, bias=False))
