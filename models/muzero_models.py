from torch import nn
from models.resnet import ResNetBasicBlock, ResNetLayer, activation_func

class representationModel(nn.Module):
    """
    
    """
    def __init__(self, in_channels=3, blocks_sizes=[128, 256, 256], deepths=[2,3,3], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=3, stride=2, padding=1, bias=False),
            activation_func(activation)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,*args, **kwargs),   # [128,48,48]
            nn.Sequential(
                nn.Conv2d(self.blocks_sizes[0], self.blocks_sizes[1], kernel_size=3, stride=2, padding=1, bias=False),
                activation_func(activation)
                ),                                      # [256,24,24]
            ResNetLayer(blocks_sizes[1], blocks_sizes[2], n=deepths[1], activation=activation, 
                        block=block,*args, **kwargs),   # [256,48,48]
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), # [256,12,12]
            ResNetLayer(blocks_sizes[2], blocks_sizes[2], n=deepths[2], activation=activation, 
                        block=block,*args, **kwargs),   # [256,12,12]
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), # [256,6,6]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class dynamicsModel(nn.Module):
    """
    
    """
    def __init__(self, in_features, blocks_size=256, depths=[4,4], 
                 activation='relu', block=ResNetBasicBlock,*args, **kwargs):
        super().__init__()

        self.layer1 = ResNetLayer(blocks_size + 1, blocks_size, n=depths[0], activation=activation, 
                        block=block,*args, **kwargs)

        self.reward_enc = nn.Sequential(nn.Conv2d(blocks_size, blocks_size, kernel_size=3, padding=1, stride=1 ,bias=False),
                                        activation_func(activation))
        self.fc_reward = nn.Linear(in_features, 1, bias=False)

        self.layer2 = ResNetLayer(blocks_size    , blocks_size, n=depths[1], activation=activation, 
                        block=block,*args, **kwargs)
        
    def forward(self, x):

        x = self.layer1(x)
        
        next_hidden_state = self.layer2(x)
        x = self.reward_enc(x)
        x = x.view(x.size(0), -1)
        reward = self.fc_reward(x)

        return next_hidden_state, reward

class predictionModel(nn.Module):
    def __init__(self, in_features, in_channels=3, num_convs=1, categorical_dim=12, 
                 activation='relu', *args, **kwargs):
        super().__init__()
        self.enc = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1 ,bias=False ,*args, **kwargs), 
            activation_func(activation)) for _ in range(num_convs)]
        )
        
        self.fc_value = nn.Linear(in_features, 1, bias=False)
        self.fc_policy = nn.Linear(in_features, categorical_dim, bias=False)
    
    def forward(self, x):
        x = self.enc(x)
        x = x.view(x.size(0), -1)
        v = self.fc_value(x)
        pi = self.fc_policy(x)

        return v, nn.functional.log_softmax(pi, dim=1)