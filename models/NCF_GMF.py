import torch
import torch.nn as nn

class NCF_GMF(nn.Module):

    def __init__(self, num_users, num_items, mlp_emb_size, gmf_emb_size):
        super(NCF_GMF, self).__init__()
        self.name = "NCF_GMF"
        self.num_users = num_users
        self.num_items = num_items
        self.mlp_emb_size = mlp_emb_size
        self.gmf_emb_size = gmf_emb_size

        self.mlp_user_emb = nn.Embedding(self.num_users, self.mlp_emb_size)
        self.mlp_item_emb = nn.Embedding(self.num_items, self.mlp_emb_size)
        self.gmf_user_emb = nn.Embedding(self.num_users, self.gmf_emb_size)
        self.gmf_item_emb = nn.Embedding(self.num_items, self.gmf_emb_size)

        self.mlp_linear = nn.Sequential(
            nn.Linear(self.mlp_emb_size * 2, 64),
            # nn.Linear(64, 32),
            # nn.Linear(32, 16),
            # nn.Linear(16, 8),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(64 + self.gmf_emb_size, 1),
            nn.Sigmoid()
        )

    
    def forward(self, users, items):
        mlp_user_emb = self.mlp_user_emb(users)
        mlp_item_emb = self.mlp_item_emb(items)
        mlp_user_item = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        mlp_output = self.mlp_linear(mlp_user_item)
        
        gmf_user_emb = self.gmf_user_emb(users)
        gmf_item_emb = self.gmf_item_emb(items)
        gmf_user_item = torch.mul(gmf_user_emb, gmf_item_emb)

        mlp_gmf = torch.cat([mlp_output, gmf_user_item], dim=-1)
        output = self.final(mlp_gmf)
        return output

