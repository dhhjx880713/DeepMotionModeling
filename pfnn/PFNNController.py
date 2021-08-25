from .pfnn import PFNN


class Trajectory(object):
    pass




class PFNNController(object):
    def __init__(self, input_dims, output_dims, ):
        
        self.model = PFNN(4, input_dim=input_dims, output_dim=output_dims)


    def load_model(self, model_file):
        self.model.load_model()
    
