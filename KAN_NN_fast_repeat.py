"""
Enables wid KAN NNs using einsum matrix computation!
"""
import torch
import torch.nn as nn
import math 

class Linear(nn.Module):
    def __init__(self, d, o, i):
        """
            d: Numer of Matrices, i.e in_dim * out_dim for hidden
            o: output_neurons
            i: input_neurons 
        """
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.empty(d, o, i))
        self.bias = nn.Parameter(torch.empty(d, o))
        self.reset_parameters(i)

    def reset_parameters(self, fan_in):
        # Match nn.Linear behavior
        bound = 1 / math.sqrt(fan_in)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # ReLU nonlinearity
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self,x):
        return torch.einsum('doi, Bdi -> Bdo', self.weight, x) + self.bias.unsqueeze(0)
    
class Output_Linear(nn.Module):
    def __init__(self, o,i,h):
        """ 
            o : out dim of the layer
            i : in_dim of the layer
            h: Number of Neurons (per univariate NN) in the final hidden layer
        """
        super(Output_Linear, self).__init__()
        self.o = o
        self.i = i 
        self.h = h
        self.weight = nn.Parameter(torch.empty(o, i * h))
        self.bias = nn.Parameter(torch.empty(o))
        self.reset_parameters(i * h)

    def reset_parameters(self, fan_in):
        bound = 1 / math.sqrt(fan_in)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        Bacth_size = x.shape[0]
        x_view = x.view(Bacth_size, self.o, self.i * self.h)
        return (torch.einsum('oi,boi->bo', self.weight, x_view) + self.bias.unsqueeze(0)).unsqueeze(-1)


class KAN_layer(nn.Module):
    """ Input and output layers are somewhat different"""
    
    def __init__(self, in_dim, out_dim, hidden):
        super(KAN_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = [1] + hidden
        self.init_layers()
    
    def init_layers(self):
        self.layers = nn.Sequential()
        for i in range(1, len(self.hidden)):
            self.layers.append(Linear(self.out_dim * self.in_dim, self.hidden[i], self.hidden[i - 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(Output_Linear(self.out_dim, self.in_dim, self.hidden[-1]))


    def forward(self, x):
        x_new = x.repeat(1,self.out_dim,1)
        #x_expanded = x.expand(x.shape[0], self.in_dim, self.out_dim)
        # Reshape to [B, in_dim * out_dim, n]
        #x_expanded = x_expanded.reshape(x.shape[0], self.in_dim * self.out_dim, 1)
        return self.layers(x_new)

    
class Neural_Kan(nn.Module):
    """
    Class:
    shape: list, describing tuple (n_1,...,n_N)
    h: shape of univariate Neural Networks. 
    """
    def __init__(self, shape, h, device = None):
        super(Neural_Kan, self).__init__()
        self.train_loss = []
        self.test_loss = []
        self.layers = nn.Sequential()
        for i in range(len(shape) - 1):
            self.layers.append(KAN_layer(in_dim = shape[i], out_dim = shape[i + 1], hidden = h))

    def forward(self,x):
        return self.layers(x).squeeze().unsqueeze(-1)
    
if __name__ == '__main__':
    model = Neural_Kan(shape = [2,4,1], h = [16])
    x = torch.randn(1, 2, 1) #### [Bacth_size, In_dim, neurons] neurons usually 1!
    print(model(x).shape)

