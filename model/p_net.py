import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from signal import signal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as Dist
from model.mlp import MLP
from utils.utils import free_mass_centre
from utils.model_init import weight_init

class P_network(nn.Module):
    def __init__(self,args):
        super(P_network,self).__init__()
        self.state_dim=args.state_dim=args.num_atoms*4
        self.num_atoms=args.num_atoms
        self.feature_dim=args.feature_dims
        self.log_sig_min=args.log_sig_min
        self.log_sig_max=args.log_sig_max

        self.mlp=MLP({"layer_sizes":[self.state_dim]+[args.latent_size]*args.layer_num_p,
                        "lr":args.learning_rate_p,"dropout":args.dropout_rate})
        self.mu_layer=nn.Linear(args.latent_size,self.num_atoms*3)
        self.log_sigma_layer=nn.Linear(args.latent_size,self.num_atoms*3)
        
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.to(self.device)
        self.apply(weight_init)
        
    def forward(self,state=None):
        if state is None:
            state=torch.zeros(self.state_dim).to(self.device)
        a=self.mlp(state)
        mu=self.mu_layer(a)
        log_sigma=self.log_sigma_layer(a)
        log_sigma=torch.clamp(log_sigma,min=self.log_sig_min,max=self.log_sig_max)
        return mu,log_sigma

    def sample(self):
        state=torch.zeros(self.state_dim).to(self.device)
        mu,log_sigma=self.forward(state)
        sigma=log_sigma.exp()
        normal=Dist.Normal(mu,sigma)
        action=normal.sample()
        log_prob=normal.log_prob(action).sum(0,keepdim=True)
        action=torch.reshape(action,(self.num_atoms,3))
        action=torch.cat((action,torch.zeros(action.shape[0],self.feature_dim-3).to(self.device)),dim=1)
        return action,log_prob
    
    def batch_sample(self,batch_size):
        state=torch.zeros(batch_size,self.state_dim).to(self.device)
        mu,log_sigma=self.forward(state)
        sigma=log_sigma.exp()
        normal=Dist.Normal(mu,sigma)
        actions=normal.sample()
        log_probs=normal.log_prob(actions).sum(1,keepdim=True)
        actions=torch.reshape(actions,(batch_size,self.num_atoms,3))
        actions=torch.cat((actions,torch.zeros(actions.shape[0],actions.shape[1],self.feature_dim-3).to(self.device)),dim=2)
        return actions,log_probs

    def sample_best_action(self):
        state=torch.zeros(self.state_dim).to(self.device)
        mu,_=self.forward(state)
        action=torch.reshape(mu,(self.num_atoms,3))
        action=torch.cat((action,torch.zeros(self.num_atoms,self.feature_dim-3).to(self.device)),dim=1)
        return action.detach()
