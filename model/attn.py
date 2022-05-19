import torch
import torch.nn as nn
import math
from torch_geometric.utils import softmax
import torch.nn.functional as F
from torch_scatter import scatter

class MetaLayer(nn.Module):
    def __init__(self,params):
        super(MetaLayer,self).__init__()
        self.emb_dim=params["emb_dim"]
        self.edge_model=(params['edge_model'] if 'edge_model' in params else None)
        self.node_model=(params['node_model'] if 'node_model' in params else None)
        self.global_model=(params['global_model'] if 'global_model' in params else None)
        self.node_attn=NodeAttn(self.emb_dim)
        self.global_edge_attn=GlobalAttn(self.emb_dim)
        self.global_node_attn=GlobalAttn(self.emb_dim)
        self.device=torch.device("cuda" if params["cuda"] else "cpu")
        self.optimizer=torch.optim.Adam(self.parameters(),lr=params["lr"])
        self.to(self.device)

    def forward(self,x,edge_index,edge_attr,u,node_batch,edge_batch,num_edge_per,num_nodes_per,num_graph):
        row=edge_index[0]
        col=edge_index[1]
        sent_attr=x[row]
        received_attr= x[col]
        # edge_model:
        global_edges = torch.repeat_interleave(u,num_edge_per,dim=0).to(self.device)
        concat_feat = torch.cat([edge_attr, sent_attr, received_attr, global_edges],dim=1).to(self.device)
        edge_attr=self.edge_model(concat_feat).to(self.device)
        sent_attr = self.node_attn(x[row], x[col], edge_attr, row, x.shape[0]).to(self.device)
        received_attr = self.node_attn(x[col], x[row], edge_attr, col, x.shape[0]).to(self.device)
        global_nodes = torch.repeat_interleave(u, num_nodes_per, dim=0).to(self.device)
        x = self.node_model(torch.cat([x, sent_attr, received_attr, global_nodes], dim=1)).to(self.device)
        # global_model & attn:
        node_attributes = self.global_node_attn(global_nodes, x, node_batch,num_graph).to(self.device)
        edge_attributes = self.global_edge_attn(global_edges,edge_attr,edge_batch,num_graph).to(self.device)
        global_feat=torch.cat([u, node_attributes, edge_attributes], dim=-1).to(self.device)
        u = self.global_model(global_feat)
        
        return x, edge_attr, u


class NodeAttn(nn.Module):
    def __init__(self,emb_dim,num_heads=None):
        super(NodeAttn,self).__init__()
        self.emb_dim=emb_dim
        if num_heads is None:
            num_heads=emb_dim//64
        self.num_heads=num_heads
        assert self.emb_dim % self.num_heads == 0  # 为什么？
        self.w1=nn.Linear(3*emb_dim,emb_dim)
        self.w2=nn.Parameter(torch.zeros((self.num_heads, self.emb_dim // self.num_heads)))
        self.w3 = nn.Linear(2 * emb_dim, emb_dim)
        self.head_dim = self.emb_dim // self.num_heads
        self.reset_parameters()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w2, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w3.weight, gain=1 / math.sqrt(2))

    def forward(self, q, k_v, k_e, index, nnode):
        """
        q: [N, C]
        k: [N, 2*c]
        v: [N, 2*c]
        """
        x = torch.cat([q, k_v, k_e], dim=1).to(self.device)
        x = self.w1(x).view(-1, self.num_heads, self.head_dim)
        x = F.leaky_relu(x)
        attn_weight = torch.einsum("nhc,hc->nh", x, self.w2).unsqueeze(-1)
        attn_weight = softmax(attn_weight, index)

        v = torch.cat([k_v, k_e], dim=1)
        v = self.w3(v).view(-1, self.num_heads, self.head_dim)
        x = (attn_weight * v).reshape(-1, self.emb_dim)
        x = scatter(x, index, dim=0, reduce="sum", dim_size=nnode)
        return x


class GlobalAttn(nn.Module):
    def __init__(self, emb_dim, num_heads=None):
        super().__init__()
        self.emb_dim = emb_dim
        if num_heads is None:
            num_heads = emb_dim // 64
        self.num_heads = num_heads
        assert self.emb_dim % self.num_heads == 0
        self.w1 = nn.Linear(2 * emb_dim, emb_dim)
        self.w2 = nn.Parameter(torch.zeros(self.num_heads, self.emb_dim // self.num_heads))
        self.head_dim = self.emb_dim // self.num_heads
        self.reset_parameter()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w2, gain=1 / math.sqrt(2))

    def forward(self, q, k, index,dim_size):
        x = torch.cat([q, k], dim=1).to(self.device)
        x = self.w1(x).view(-1, self.num_heads, self.head_dim)
        x = F.leaky_relu(x)
        attn_weight = torch.einsum("nhc,hc->nh", x, self.w2)
        attn_weight = softmax(attn_weight, index,dim=0).unsqueeze(-1)

        v = k.view(-1, self.num_heads, self.head_dim)
        x = (attn_weight * v).reshape(-1, self.emb_dim)
        x = scatter(x, index, dim=0, reduce="sum", dim_size=dim_size)
        return x
        