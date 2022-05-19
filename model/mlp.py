import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_init import weight_init

class MLP(nn.Module):
    def __init__(self,params):
        super(MLP,self).__init__()
        self.layer_sizes=params['layer_sizes']
        self.activation=(params['activation'] if 'activation' in params else nn.ReLU(inplace=False))
        self.dropout=(params['dropout'] if 'dropout' in params else 0)
        self.use_layer_norm=(params['use_layer_norm'] if 'use_layer_norm' in params else False)
        self.layer_norm_before=(params['layer_norm_before'] if 'layer_norm_before' in params else False)
        self.use_batch_norm=(params['use_batch_norm'] if 'use_batch_norm' in params else False)
        self.batch_norm=(params['batch_norm'] if 'batch_norm' in params else nn.BatchNorm1d)
        self.module_list=nn.ModuleList()
        if not self.use_batch_norm:
            if self.layer_norm_before:
                self.module_list.append(nn.LayerNorm(self.layer_sizes[0]))
            for i in range(1,len(self.layer_sizes)):
                self.module_list.append(nn.Linear(self.layer_sizes[i-1],self.layer_sizes[i]))
                if(self.layer_sizes[i]>1):
                    self.module_list.append(self.activation)
                    if self.dropout>0:
                        self.module_list.append(nn.Dropout(self.dropout))
            if not self.layer_norm_before and self.use_layer_norm:
                self.module_list.append(nn.LayerNorm(self.layer_sizes[-1]))
        else:
            for i in range(1,len(self.layer_sizes)):
                self.module_list.append(nn.Linear(self.layer_sizes[i-1],self.layer_sizes[i]))
                if(self.layer_sizes[i]>1):
                    self.module_list.append(self.batch_norm(self.layer_sizes[i]))
                    self.module_list.append(self.activation)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer=torch.optim.Adam(self.parameters(),lr=params['lr'])
        self.reset_parameters()
        self.to(self.device)
        
        self.apply(weight_init)

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self,x):
        x.to(self.device)
        for i,module in enumerate(self.module_list):
            x=module(x)
        return x

class MLP2D(nn.Module):
    def __init__(self,params):
        super(MLP2D,self).__init__()
        self.output_size=params["layer_sizes"][-1]
        params["layer_sizes"]=self.flatten_layer_sizes(params["layer_sizes"])
        self.mlp=MLP(params)

    
    def flatten_layer_sizes(self,layer_sizes_2d):
        layer_sizes=[]
        for i in range(len(layer_sizes_2d)):
            layer_sizes.append(layer_sizes_2d[i][0]*layer_sizes_2d[i][1])
        return layer_sizes

    def flatten_input(self,x):
        return torch.flatten(x)
    
    def restore_output(self,x):
        return torch.reshape(x,self.output_size)

    def forward(self,x):
        x=self.flatten_input(x)
        x=self.mlp.forward(x)
        x=self.restore_output(x)
        return x

class DropoutIfTraining(nn.Module):
    """
    Borrow this implementation from deepmind
    """

    def __init__(self, p=0.0, submodule=None):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = p
        self.submodule = submodule if submodule else nn.Identity()

    def forward(self, x):
        x = self.submodule(x)
        newones = x.new_ones((x.size(0), 1))
        newones = F.dropout(newones, p=self.p, training=self.training,inplace=False)
        out = x * newones
        return out
