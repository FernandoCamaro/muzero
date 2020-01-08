from network import videogameNetwork
from models.muzero_models import representationModel, dynamicsModel, predictionModel

def MarioNet(action_space_size: int, cuda: bool=True):

    # observation [H,W] = [96,96]
    in_channels = int((1+3)*4) # 4 rgb images with their respective previous actions 
    rep_bs = [128, 256, 256]
    rep_model = representationModel(in_channels=in_channels, blocks_sizes=rep_bs, deepths=[2,3,3], activation='relu')
    in_features = int(256*6*6)
    dyn_model = dynamicsModel(in_features=in_features, blocks_size=rep_bs[-1], depths=[4,4], activation='relu')
    pred_model = predictionModel(in_features=in_features, in_channels=rep_bs[-1], num_convs=1, categorical_dim=action_space_size, activation='relu')

    net = videogameNetwork(rep_model, dyn_model, pred_model, action_space_size, cuda=cuda)

    return net