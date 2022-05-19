import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from model.attn import MetaLayer
from model.mlp import MLP,DropoutIfTraining
from utils.utils import get_random_rotation
from utils.model_init import weight_init

class Q_network(nn.Module):
    def __init__(self,args):
        super(Q_network,self).__init__()
        self.atom_type_num=args.atom_type_num
        self.edge_type_num=args.edge_type_num
        self.layer_num=args.layer_num_q
        self.mlp_hidden_layer_size=args.mlp_hidden_layer_size
        self.latent_size=args.latent_size
        self.message_pass_steps=args.message_pass_num
        self.dropout_rate=args.dropout_rate
        self.node_encoder=MLP({"layer_sizes":[self.atom_type_num]+[self.mlp_hidden_layer_size]*self.layer_num+[self.latent_size],
                                "use_layer_norm":True,"use_batch_norm":True,"lr":args.learning_rate_q})
        self.edge_encoder=MLP({"layer_sizes":[self.edge_type_num]+[self.mlp_hidden_layer_size]*self.layer_num+[self.latent_size],
                                "use_layer_norm":True,"use_batch_norm":True,"lr":args.learning_rate_q})
        self.pos_embedding=MLP({"layer_sizes":[3,self.latent_size,self.latent_size,self.latent_size],"lr":args.learning_rate_q})
        self.dis_embedding=MLP({"layer_sizes":[1,self.latent_size,self.latent_size,self.latent_size],"lr":args.learning_rate_q})
        self.gn_block_list=nn.ModuleList()
        for i in range(self.message_pass_steps):
            edge_model=DropoutIfTraining(
                p=args.dropout_rate,
                submodule=MLP({"layer_sizes":[self.latent_size*4]+[self.mlp_hidden_layer_size]*self.layer_num+[self.latent_size],
                                "use_layer_norm":True,"use_batch_norm":True,"lr":args.learning_rate_q,
                                "layer_norm_before":True,"dropout":args.dropout_rate})
            )
            node_model=DropoutIfTraining(
                p=args.dropout_rate,
                submodule=MLP({"layer_sizes":[self.latent_size*4]+[self.mlp_hidden_layer_size]*self.layer_num+[self.latent_size],
                                "use_layer_norm":True,"use_batch_norm":True,"lr":args.learning_rate_q,
                                "layer_norm_before":True,"dropout":args.dropout_rate})
            )
            global_model=DropoutIfTraining(
                p=args.dropout_rate,
                submodule=MLP({"layer_sizes":[self.latent_size*3]+[self.mlp_hidden_layer_size]*self.layer_num+[self.latent_size],
                                "use_layer_norm":True,"use_batch_norm":True,"lr":args.learning_rate_q,
                                "layer_norm_before":True,"dropout":args.dropout_rate})
            )
            meta_layer_params={"emb_dim":self.latent_size,
                                "edge_model":edge_model,
                                "node_model":node_model,
                                "global_model":global_model,
                                "lr":args.learning_rate_q,
                                "cuda":args.cuda,}
            self.gn_block_list.append(MetaLayer(meta_layer_params))
        self.global_decoder=MLP({"layer_sizes":[self.latent_size]+[self.mlp_hidden_layer_size]*self.layer_num+[1],
                                "lr":args.learning_rate_q})
        self.device=torch.device("cuda" if args.cuda else "cpu")
        self.to(self.device)
        self.apply(weight_init)

    def actions_to_graphs_and_flatten(self,actions):
        n_graph=actions.shape[0]
        edge_attr=torch.zeros((actions.shape[0],actions.shape[1]*(actions.shape[1]-1),1),dtype=torch.float).to(self.device)
        edge_index=torch.zeros(2,actions.shape[0],actions.shape[1]*(actions.shape[1]-1),dtype=torch.long).to(self.device)
        x=actions[:,:,3:].to(self.device)
        x=x.view(1,-1,x.shape[2])
        x=x.view(x.shape[1],x.shape[2])
        edge_attr=edge_attr.view(1,-1,edge_attr.shape[2])
        edge_attr=edge_attr.view(edge_attr.shape[1],edge_attr.shape[2])
        pos=actions[:,:,0:3].to(self.device)
        pos=pos.view(1,-1,pos.shape[2])
        pos=pos.view(pos.shape[1],pos.shape[2])
        for index in range(actions.shape[0]):
            ptr=0
            for m in range(actions.shape[1]):
                for n in range(actions.shape[1]):
                    if m!=n:
                        edge_index[0][index][ptr]=m+index*actions.shape[1]
                        edge_index[1][index][ptr]=n+index*actions.shape[1]
                        ptr=ptr+1
        edge_index=edge_index.view(2,edge_index.shape[1]*edge_index.shape[2])
        graph_batch=torch.arange(n_graph).to(self.device)
        node_batch=torch.repeat_interleave(graph_batch,actions.shape[1],dim=0).to(self.device)
        edge_batch=torch.repeat_interleave(graph_batch,actions.shape[1]*(actions.shape[1]-1),dim=0).to(self.device)
        return Data(x=x,edge_attr=edge_attr,edge_index=edge_index,pos=pos,node_batch=node_batch,edge_batch=edge_batch)

    def forward(self, actions):
        n_graph=actions.shape[0]
        num_edge_per=actions.shape[1]*(actions.shape[1]-1)
        num_nodes_per=actions.shape[1]
        graphs=self.actions_to_graphs_and_flatten(actions)
        x=graphs.x.to(self.device)
        edge_attr=graphs.edge_attr.to(self.device)
        edge_index=graphs.edge_index.to(self.device)
        pos=graphs.pos.to(self.device)
        node_batch=graphs.node_batch.to(self.device)
        edge_batch=graphs.edge_batch.to(self.device)
        assert x.shape[1]==1
        node_attr=F.one_hot(x.to(torch.long).view(-1),self.atom_type_num).to(torch.float).to(self.device)
        assert edge_attr.shape[1]==1
        edge_attr=F.one_hot(edge_attr.to(torch.long).view(-1),self.edge_type_num).to(torch.float).to(self.device)
        global_r=nn.Parameter(torch.zeros((n_graph, self.latent_size),dtype=torch.float32)).to(self.device)
        node_attr_embed=self.node_encoder(node_attr).to(self.device)
        edge_attr_embed=self.edge_encoder(edge_attr).to(self.device)
        for i,layer in enumerate(self.gn_block_list):
            node_attr_embed,edge_attr_embed=self.positional_embedding(
                                                    node_attr_embed,
                                                    edge_attr_embed,
                                                    pos,edge_index)
            node_attr_embed.to(self.device)
            edge_attr_embed.to(self.device)
            node_attr_prime,edge_attr_prime,global_r_prime=layer(node_attr_embed,edge_index,edge_attr_embed,global_r,
                                                                node_batch,edge_batch,num_edge_per,num_nodes_per,n_graph)
            node_attr_embed=F.dropout(node_attr_prime,p=self.dropout_rate,training=self.training)+node_attr_embed.clone()
            edge_attr_embed=F.dropout(edge_attr_prime,p=self.dropout_rate,training=self.training)+edge_attr_embed.clone()
            global_r=F.dropout(global_r_prime,p=self.dropout_rate,training=self.training)+global_r.clone()
            pos=get_random_rotation(pos)
        r=self.global_decoder(global_r).to(self.device)
        return r

    def positional_embedding(self,node_attr_embed,edge_attr_embed,pos,edge_index):
        node_attr_embed=node_attr_embed+self.pos_embedding(pos)
        row = edge_index[0]
        col = edge_index[1]
        sent_pos = pos[row]
        received_pos = pos[col]
        length = (sent_pos - received_pos).norm(dim=-1).unsqueeze(-1)
        edge_attr_embed = edge_attr_embed + self.dis_embedding(length)
        return node_attr_embed,edge_attr_embed