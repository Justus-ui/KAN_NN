import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured
import time

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, mask):
        super(Linear, self).__init__()
        self.Lin_layer = nn.Linear(in_dim, out_dim)
        self.mask = mask

    def forward(self, x):
        masked_weights = (self.Lin_layer.weight * self.mask)#.to_sparse()
        return F.linear(x, masked_weights, self.Lin_layer.bias)


class SparseNeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, h = [8,4]):
        super(SparseNeuralNetwork, self).__init__()
        h = h
        self.univariate_nn = nn.Sequential()
        layers = []
        self.masks = []
        self.num_funcs = 3
        for layer in range(len(h)):
            if layer == 0:
                ### INput layer need special mask computation
                dim_in = in_dim 
                dim_out = in_dim* out_dim * h[layer]
                self.masks.append(self.hidden_sparistiy_masks(dim_in, dim_out, [1, out_dim *h[layer]]))
                mask = self.hidden_sparistiy_masks(dim_in, dim_out, [1, out_dim *h[layer]])
                layers.append(Linear(dim_in, dim_out, mask))
                #layers.append(nn.BatchNorm1d(h[layer] * in_dim, affine=True))
                layers.append(nn.ReLU())
                continue
            dim_in = in_dim * out_dim * h[layer-1]
            dim_out = in_dim * out_dim * h[layer]
            self.masks.append(self.hidden_sparistiy_masks(dim_in, dim_out , h = [h[layer-1], h[layer]]))
            mask = self.hidden_sparistiy_masks(dim_in, dim_out , h = [h[layer-1], h[layer]])
            layers.append(Linear(dim_in, dim_out, mask))
            #layers.append(nn.BatchNorm1d(h[layer] * in_dim, affine=True))
            layers.append(nn.ReLU())

        self.univariate_nn = nn.Sequential(*layers)
        #self.multiply_weight_masks()
        mask = self.out_layer_mask(in_dim* out_dim * h[-1], out_dim, [h[-1],1], input_dimension = in_dim)
        self.masks.append(mask)
        self.fc2 = Linear(in_dim* out_dim * h[-1], out_dim, mask)

    def out_layer_mask(self, in_dim, out_dim, h, input_dimension):
        mask = torch.zeros(out_dim, in_dim)
        output_neurons = h[1]
        input_neurons = h[0]
        step = out_dim * h[0]
        for i in range(0,in_dim):
            for j in range(input_dimension):
                mask[i*output_neurons:output_neurons*(i + 1) , i*input_neurons + (j*step):((i + 1)*input_neurons) + (j*step)] = 1
        return mask

    def hidden_sparistiy_masks(self, in_dim, out_dim, h):
        mask = torch.zeros(out_dim, in_dim)
        output_neurons = h[1]
        input_neurons = h[0]
        for i in range(0,in_dim):
            mask[i*output_neurons:output_neurons*(i + 1) , i*input_neurons:(i + 1)*input_neurons] = 1
        return mask

    def forward(self, x):
        hidden = self.univariate_nn(x)
        output = self.fc2(hidden)
        return output

class Neural_Kan(nn.Module):
    """
    Class:
    shape: list, describing tuple (n_1,...,n_N)
    h: shape of univariate Neural Networks. 
    """
    #TODO Extend h to be layerwise NN (Lipschitz of first Layer is not so bad!)
    def __init__(self, shape, h):
        super(Neural_Kan, self).__init__()
        self.train_loss = []
        self.test_loss = []
        self.layers = nn.Sequential()
        self.params = 0
        for i in range(len(shape) - 1):
            model = SparseNeuralNetwork(in_dim = shape[i], out_dim = shape[i + 1], h = h)
            self.layers.append(model)
            self.params += sum(mask.count_nonzero().item() for mask in model.masks)
        #print(self)


    def forward(self,x):
        return self.layers(x)

if __name__ == "__main__":
    import time
    #model = torch.jit.script(Neural_Kan(shape = [12,12,3], h = [8]))
    model = Neural_Kan(shape = [2,13,1,1], h = [32])
    criterion = torch.nn.MSELoss()
    start_time = time.time()
    output = model(torch.randn(100,2))
    target = torch.randn_like(output)
    loss = criterion(output, target)
    loss.backward()
    print(model.params)
    print("Elapsed time:",time.time()-start_time)
        
