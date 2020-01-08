from network import videogameNetwork
from models.muzero_models import representationModel, dynamicsModel, predictionModel

def MarioNet():
    
    # observation [H,W] = [96,96]
    in_channels = int((1+3)*4) # 4 rgb images with their respective previous actions 
    rep_bs = [128, 256, 256]
    num_actions = 12
    rep_model = representationModel(in_channels=in_channels, blocks_sizes=rep_bs, deepths=[2,3,3], activation='relu')
    dyn_model = dynamicsModel(blocks_size=rep_bs[-1], depth=8, activation='relu')
    in_features = int(256*6*6)
    pred_model = predictionModel(in_features=in_features, in_channels=rep_bs[-1], num_convs=1, categorical_dim=num_actions, activation='relu')

    net = videogameNetwork(rep_model, dyn_model, pred_model, num_actions, cuda=True)

    return net