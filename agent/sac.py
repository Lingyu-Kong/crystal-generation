import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
import random
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from model.p_net import P_network
from model.gnn import Q_network
import torch.distributions as Dist
from utils.utils import check_if_action_accepted, evaluate_action_by_ase, evaluate_actions_by_ase, explore_by_dft,evaluate_actions_by_ase

class Replay_Buffer():
    def __init__(self,max_size,num_atoms,feature_dims):
        self.max_size=max_size
        self.mem_cntr=0
        self.feature_dims=feature_dims
        self.num_atoms=num_atoms
        self.action_memory=np.zeros((self.max_size,self.num_atoms,self.feature_dims),dtype=np.float32)
        self.reward_memory=np.zeros((self.max_size,1),dtype=np.float32)

    def store_transition(self,action,reward):
        index=self.mem_cntr%self.max_size
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.mem_cntr+=1

    def sample_buffer(self,batch_size):
        # self.random_shuffle()
        max_mem=min(self.mem_cntr,self.max_size)
        batch=np.random.choice(max_mem,size=batch_size)
        batch_actions=self.action_memory[batch]
        batch_rewards=self.reward_memory[batch]
        return batch_actions,batch_rewards

    # def read_buffer(self):
    #     return self.action_memory,self.reward_memory,min(self.mem_cntr,self.max_size)

    def random_shuffle(self):
        max_mem=min(self.mem_cntr,self.max_size)
        index=np.arange(max_mem)
        shuffle=np.random.permutation(index)
        self.action_memory[index]=self.action_memory[shuffle]
        self.reward_memory[index]=self.reward_memory[shuffle]


class SAC(object):
    def __init__(self,args):

        self.alpha=args.temperature
        self.automatic_entropy_tuning=args.automatic_entropy_tuning
        self.replay_buffer=Replay_Buffer(args.buffer_size,args.num_atoms,args.feature_dims)

        self.device=torch.device("cuda" if args.cuda else "cpu")
        self.batch_size=args.batch_size
        self.num_atoms=args.num_atoms
        self.feature_dims=args.feature_dims
        self.explore_steps=args.explore_steps

        self.p_net=P_network(args)
        self.q_net=Q_network(args)

        if self.automatic_entropy_tuning:
            self.target_entropy=-torch.prod(torch.Tensor([args.num_atoms,3]).to(self.device)).item()
            self.log_alpha=torch.zeros(1,requires_grad=True,device=self.device)
            self.alpha_optim=Adam([self.log_alpha],lr=args.learning_rate_t)
        
        # self.replay_buffer_init(args.replay_buffer_init_size,args.num_atoms)

    def train_one_step(self):
        batch_actions,batch_rewards=self.replay_buffer.sample_buffer(self.batch_size)
        batch_actions=torch.FloatTensor(batch_actions).to(self.device)
        batch_rewards=torch.FloatTensor(batch_rewards).to(self.device)
        # train q_net
        q_values=self.q_net(batch_actions).to(self.device)
        q_loss=F.mse_loss(q_values,batch_rewards)/self.batch_size
        self.q_net.optim.zero_grad()
        q_loss.backward()
        self.q_net.optim.step()
        # train p_net
        log_probs=self.p_net.get_log_probs(batch_actions)
        p_loss=((self.alpha*log_probs)-batch_rewards).mean()
        # actions,log_probs=self.p_net.batch_sample(self.batch_size)
        # q_values=self.q_net(actions).to(self.device)
        # p_loss=((self.alpha * log_probs) - q_values).mean()
        # self.p_net.optim.zero_grad()
        # p_loss.backward()
        # self.p_net.optim.step()
        # tune temperature
        if self.automatic_entropy_tuning:
            alpha_loss=-(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha=self.log_alpha.exp()
        predicted_best_reward=torch.max(q_values).item()
        actions,_=self.p_net.batch_sample(self.batch_size)
        real_q_values=evaluate_actions_by_ase(actions).to(self.device)
        real_best_reward=real_q_values.mean().item()
        return q_loss.item(),p_loss.item(),predicted_best_reward,real_best_reward,log_probs.mean(),q_values.mean(),self.alpha
    
    def replay_buffer_init(self,replay_buffer_init_size):
        ctr=0
        while ctr<replay_buffer_init_size:
            pos=torch.rand(self.num_atoms,3)*5
            if check_if_action_accepted(pos):
                action=torch.cat((pos,torch.zeros(self.num_atoms,self.feature_dims-3)),dim=1)
                action,reward=explore_by_dft(action,int(self.explore_steps*random.random())+1)
                action=action.detach().cpu().numpy()
                self.replay_buffer.store_transition(action,reward)
                ctr+=1

    # def replay_buffer_init(self,replay_buffer_init_size):
    #     ctr=0
    #     while ctr<replay_buffer_init_size:
    #         pos=torch.rand(self.num_atoms,3)*5
    #         if check_if_action_accepted(pos):
    #             action=torch.cat((pos,torch.zeros(self.num_atoms,self.feature_dims-3)),dim=1)
    #             reward=evaluate_action_by_ase(action)
    #             action=action.detach().cpu().numpy()
    #             self.replay_buffer.store_transition(action,reward)
    #             ctr+=1

    def expand_buffer(self,expand_buffer_size):
        actions,_=self.p_net.batch_sample(expand_buffer_size)
        for i in range(expand_buffer_size):
            reward=evaluate_action_by_ase(actions[i])
            action=actions[i].detach().cpu()
            self.replay_buffer.store_transition(action.numpy(),reward)
            explored_action,explored_reward=explore_by_dft(action,int(self.explore_steps*random.random())+1)
            explored_action=explored_action.detach().cpu().numpy()
            self.replay_buffer.store_transition(explored_action,explored_reward)
        