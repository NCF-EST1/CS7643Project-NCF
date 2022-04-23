import torch
import torch.nn as nn

class NeuMF(nn.Module):

    def __init__(self, num_users, num_items, mlp_emb_size, gmf_emb_size, layers):
        super(NeuMF, self).__init__()
        self.name = "NeuMF"
        self.num_users = num_users
        self.num_items = num_items
        self.mlp_emb_size = mlp_emb_size
        self.gmf_emb_size = gmf_emb_size
        self.layers = layers

        self.mlp_user_emb = nn.Embedding(self.num_users, self.mlp_emb_size)
        self.mlp_item_emb = nn.Embedding(self.num_items, self.mlp_emb_size)
        self.gmf_user_emb = nn.Embedding(self.num_users, self.gmf_emb_size)
        self.gmf_item_emb = nn.Embedding(self.num_items, self.gmf_emb_size)

        self.mlp_linear = nn.ModuleList()
        self.mlp_linear.append(nn.Linear(self.mlp_emb_size*2, layers[0]))
        self.mlp_linear.append(nn.ReLU())
        for i in range(1, len(layers)):
            self.mlp_linear.append(nn.Linear(layers[i-1], layers[i]))
            self.mlp_linear.append(nn.ReLU())

        self.final = nn.Sequential(
            nn.Linear(self.layers[-1] + self.gmf_emb_size, 1),
            nn.Sigmoid()
        )

    
    def forward(self, users, items):
        mlp_user_emb = self.mlp_user_emb(users)
        mlp_item_emb = self.mlp_item_emb(items)
        mlp_user_item = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        mlp_linear_tensor = mlp_user_item
        for i in range(len(self.mlp_linear)):
            mlp_linear_tensor = self.mlp_linear[i](mlp_linear_tensor) 
        
        gmf_user_emb = self.gmf_user_emb(users)
        gmf_item_emb = self.gmf_item_emb(items)
        gmf_user_item = torch.mul(gmf_user_emb, gmf_item_emb)

        mlp_gmf = torch.cat([mlp_linear_tensor, gmf_user_item], dim=-1)
        output = self.final(mlp_gmf)
        return output

