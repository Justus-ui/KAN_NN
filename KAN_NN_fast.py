"""
    Implements the KANN using einsum instead of Sparse Linear Layers, use when KANN shape contains large values, for saving memory, performs slightly worse than Linear during optimization
"""

import torch
import torch.nn as nn
import math 
import opt_einsum as oe

class Linear(nn.Module):
    def __init__(self, d, o, i):
        """
            d: Numer of Matrices, i.e in_dim * out_dim for hidden
            o: output_neurons
            i: input_neurons a
        """
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.empty(d, o, i))
        self.bias = nn.Parameter(torch.empty(d, o))
        self.reset_parameters(i)

    def reset_parameters(self, fan_in):
        nn.init.kaiming_uniform_(self.weight, a=0, nonlinearity='relu')
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self,x):
        #return torch.einsum('doi, Bdi -> Bdo', self.weight, x) + self.bias.unsqueeze(0)
        return oe.contract('doi, Bdi -> Bdo', self.weight, x, optimize='greedy') + self.bias.unsqueeze(0)
    
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
        nn.init.kaiming_uniform_(self.weight, a=0, nonlinearity='relu')
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        Bacth_size = x.shape[0]
        x_view = x.reshape(Bacth_size, self.o, self.i * self.h)
        #return torch.einsum('oi,boi->bo', self.weight, x_view) + self.bias.unsqueeze(0)
        return (oe.contract('oi,boi->bo', self.weight, x_view, optimize='greedy') + self.bias.unsqueeze(0))

class Input_Linear(nn.Module):
    def __init__(self, in_dim, out_dim, o):
        """
            d: Numer of Matrices, i.e in_dim * out_dim for hidden
            o: output_neurons
            i: input_neurons 
        """
        super(Input_Linear, self).__init__()
        self.d = in_dim * out_dim
        self.o = o

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim, o))
        self.bias = nn.Parameter(torch.empty(self.d, o))

        self.reset_parameters(1)

    def reset_parameters(self, fan_in):
        nn.init.kaiming_uniform_(self.weight, a=0, nonlinearity='relu')
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self,x):
        B = x.shape[0]
        out = x[:, None, :, None] * self.weight[None, :, :, :]
        return out.reshape(B, self.d, self.o) + self.bias.unsqueeze(0)
    
class KAN_layer(nn.Module):
    """ Input and output layers are somewhat different"""
    
    def __init__(self, in_dim, out_dim, hidden):
        super(KAN_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.init_layers()
    
    def init_layers(self):
        self.layers = nn.Sequential()
        self.layers.append(Input_Linear(self.in_dim, self.out_dim, self.hidden[0]))
        self.layers.append(nn.ReLU())
        for i in range(1, len(self.hidden)):
            self.layers.append(Linear(self.out_dim * self.in_dim, self.hidden[i], self.hidden[i - 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(Output_Linear(self.out_dim, self.in_dim, self.hidden[-1]))
        #self.layers.append(nn.BatchNorm1d(self.hidden[-1], affine=True))


    def forward(self, x):
        return self.layers(x)

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
            print(shape[i], shape[i + 1])
            self.layers.append(KAN_layer(in_dim = shape[i], out_dim = shape[i + 1], hidden = h))

    def forward(self,x):
        return self.layers(x)
    
if __name__ == '__main__':
    model = Neural_Kan(shape = [9,1], h = [16])
    x = torch.randn(1, 9) #### [Bacth_size, In_dim] neurons usually 1!
    print(x, "input")
    print(model(x))

