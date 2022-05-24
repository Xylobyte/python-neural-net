# Base Layer Class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Computes the output Y of a layer for a given input X
    def forward_propogation(self, input):
        raise NotImplementedError

    # Computes dE/dX for a given dE/dY (and update paramters)
    def backward_propogation(self, output_error, learning_rate):
        raise NotImplementedError