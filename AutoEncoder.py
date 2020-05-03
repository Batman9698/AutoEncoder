import torch.nn as nn
#AE_v4
class AutoEncoder_v4(nn.Module):
    def __init__(self, encoding_dim):
        super(AutoEncoder_v4, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(784, encoding_dim),
                                    nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, 784),
                                    nn.Sigmoid())
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

# AE_v3
class AutoEncoder_v3(nn.Module):
    def __init__(self, encoding_dim):
        super(AutoEncoder_v3, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 512),
                                    nn.ReLU(),
                                    nn.Linear(512,256),
                                    nn.ReLU(),
                                    nn.Linear(256, encoding_dim))
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 784),
                                    nn.Sigmoid())
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

# AE_v2
class AutoEncoder_v2(nn.Module):
    def __init__(self, encoding_dim):
        super(AutoEncoder_v2, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(784, encoding_dim),
                                    nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, 784),
                                    nn.Sigmoid())
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)