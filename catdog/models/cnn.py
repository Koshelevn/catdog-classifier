import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class ModelCNN:
    def __init__(self, embedding_size):

        self.model_cnn = nn.Sequential()

        self.model_cnn.add_module("conv_1", nn.Conv2d(3, 6, 5))
        self.model_cnn.add_module("relu_1", nn.ReLU())
        self.model_cnn.add_module("max_pooling_1", nn.MaxPool2d(2, 2))

        self.model_cnn.add_module("conv_2", nn.Conv2d(6, 16, 5))
        self.model_cnn.add_module("relu_2", nn.ReLU())
        self.model_cnn.add_module("max_pooling_2", nn.MaxPool2d(2, 2))

        self.model_cnn.add_module("global_max_pooling", nn.AdaptiveMaxPool2d(4))
        self.model_cnn.add_module("dropout", nn.Dropout(0.3))
        self.model_cnn.add_module("flat", Flatten())

        self.model_cnn.add_module("fc", nn.Linear(256, embedding_size))
        self.model_cnn.add_module("relu", nn.ReLU())

        self.model_cnn.add_module("dropout_6", nn.Dropout(0.3))

        self.model_cnn.add_module("fc_logits", nn.Linear(embedding_size, 2, bias=False))

    def get_model(self):
        return self.model_cnn
