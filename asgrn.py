# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 23:49:47 2022

@author: MSH
"""

import torch
from torch import nn
from torch.nn import Linear, ReLU, Dropout,Sigmoid
from torch_geometric.nn import Sequential, GatedGraphConv,knn_graph
from torch_geometric.utils import dropout_adj, subgraph,softmax

class seq2seq_quantile(nn.Module):
    def __init__(self, node_features, window_length,horizon,hidden_state,nquantiles, device):
        super().__init__()
        self.name = 'seq2seq_quantile'
        self.node_features = node_features
        self.n = window_length
        self.hidden_state_size = horizon * hidden_state    
        self.h = horizon
        self.nquantiles = nquantiles
        self.device = device
        assert self.hidden_state_size % 2 == 0     

        self.embedding_x = nn.Sequential(
            nn.Linear(self.node_features, self.node_features), 
            nn.Tanh(),         
        )
        
        self.embedding_z = nn.Sequential(
            nn.Linear(self.node_features + self.hidden_state_size//self.h-1, self.node_features + self.hidden_state_size//self.h-1), 
            nn.Tanh(),
         
        )   
        
        self.encoder = nn.LSTMCell(self.node_features, self.hidden_state_size)
        self.decoder = nn.LSTMCell(self.node_features + self.hidden_state_size//self.h-1, self.hidden_state_size)
        
        self.Linear0 = nn.Sequential(
            nn.Linear(self.hidden_state_size, self.hidden_state_size),            
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
        )           

        self.out = nn.Sequential(
            nn.Linear(self.hidden_state_size, self.hidden_state_size//2), 
            nn.LeakyReLU(),
            nn.Linear(self.hidden_state_size//2, self.nquantiles),
            nn.LeakyReLU(0.1),           
        )

    def forward(self, window):        
    
        
        h = torch.zeros(window.x_dict['node'].shape[0], self.hidden_state_size).to(self.device) 
        c = torch.zeros(window.x_dict['node'].shape[0], self.hidden_state_size).to(self.device)      
        
        for t in range(0,self.n):
            x = window.x_dict['node'][:,:,t] # first dim is the number of features, nodes second, and time thrid； when enther the glstm, the nodes should be the first, features second
            x = self.embedding_x(x)
            h, c = self.encoder(x,(h,c)  ) 

        h = self.Linear0(h)
        z = torch.cat([h.view(-1,self.hidden_state_size//self.h,self.h),window['node'].z],1)
        pred = []
        for t in range(self.h):
            x = self.embedding_z(z[:,:,t])
            h, c = self.decoder(x,(h,c))            
            y = self.out(h)
            pred.append(y.unsqueeze(2))
        
        pred = torch.cat(pred,2)

        return pred


    
class ASG_seq2seq_quantile_onehot(nn.Module):
    def __init__(self, node_features, window_length,horizon, attr_levels, K,hidden_state,nquantiles,device):
        super().__init__()
        self.name = 'ASG_seq2seq_quantile_onehot'
        self.node_features = node_features
        self.attr_levels = attr_levels
        self.n = window_length
        self.hidden_state_size = horizon * hidden_state
        
        self.k = K        
        self.h = horizon
        self.g_hidden = 16         
        self.nquantiles = nquantiles
        self.node_attrs = len(attr_levels)
        self.device = device
        
        assert self.hidden_state_size % 2 == 0      
       
        
        self.embedding_attr = [ nn.Sequential(
                                  nn.Linear(attr_levels[i], 1, bias = False),
                                   nn.Tanh(),)         
                                  for i in range(self.node_attrs) ]
                
                
                 
        self.embedding_x = nn.Sequential(
            nn.Linear(self.node_features, self.node_features), 
            nn.Tanh(),
         
        )
        
        self.embedding_z = nn.Sequential(
            nn.Linear(self.node_features + self.hidden_state_size//self.h-1, self.node_features + self.hidden_state_size//self.h-1), 
            nn.Tanh(),
         
        )
        
        self.gcn_more = Sequential('x, edge_index,edge_weight', [
            (GatedGraphConv(self.g_hidden,3), 'x, edge_index,edge_weight -> x'),  
            (Dropout(p=0.2), 'x -> x'),
            ( nn.LeakyReLU(), 'x -> x'),    
        ])    
        
        self.encoder = nn.LSTMCell(self.node_features, self.hidden_state_size)
        self.decoder = nn.LSTMCell(self.node_features + self.hidden_state_size//self.h-1+ self.g_hidden, self.hidden_state_size)
        
        self.Linear0 = nn.Sequential(
            nn.Linear(self.hidden_state_size, self.hidden_state_size),            
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
        )   
        
      

        self.out = nn.Sequential(
            nn.Linear(self.hidden_state_size, self.hidden_state_size//2), 
            nn.LeakyReLU(),
            nn.Linear(self.hidden_state_size//2, self.nquantiles),
            nn.LeakyReLU(0.1),        
        )

    def forward(self, window):        
    
        
        h = torch.zeros(window.x_dict['node'].shape[0], self.hidden_state_size).to(self.device) 
        c = torch.zeros(window.x_dict['node'].shape[0], self.hidden_state_size).to(self.device)
        
        attr = [torch.nn.functional.one_hot(window['node'].f[:,i].long(), num_classes= self.attr_levels[i]) for i in range(self.node_attrs)]
        
        s_l =  torch.cat( [self.embedding_attr[i].to(self.device)(attr[i].float() ) for i in range(self.node_attrs)],-1)       
               
        #edge_index_global = knn(s_l, s_l, self.k, window['node'].batch, window['node'].batch)#.flip([0])   
        edge_index_global = knn_graph(s_l, self.k, window['node'].batch,loop= True)#.flip([0])  

        for t in range(0,self.n):            
            x = self.embedding_x(window.x_dict['node'][:,:,t])
            h, c = self.encoder(x,(h,c)  ) 

        h = self.Linear0(h)
        z = torch.cat([h.view(-1,self.hidden_state_size//self.h,self.h),window['node'].z],1)
        pred = []
        
        for t in range(self.h):              
           
            edge_index,_ = subgraph(window['node'].mask[:,t].bool(),edge_index_global)
            edge_index,_ = dropout_adj(edge_index,p=0.2)            

            edge_weight = (s_l[edge_index[0]]- s_l[edge_index[1]]).pow(2).sum(-1)        
            edge_weight = torch.exp(-10. * edge_weight)  
            edge_weight = softmax(edge_weight,edge_index[1])  
            
            g = self.gcn_more(window['node'].z[:,0:4,t],edge_index,edge_weight)   
            
            x = self.embedding_z(z[:,:,t])
            h, c = self.decoder(torch.cat([x,g],dim=-1),(h,c))                   
            y = self.out(h)
            pred.append(y.unsqueeze(2))
        
        pred = torch.cat(pred,2)

        return pred,s_l