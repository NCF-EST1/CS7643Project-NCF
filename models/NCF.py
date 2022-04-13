import torch
import torch.nn as nn

class NCF(nn.Module):

    def __init__(self, num_users, num_items, emb_size, layers):
        super(NCF, self).__init__()
        self.name = "NCF"
        self.num_users = num_users
        self.num_items = num_items
        self.emb_size = emb_size
        self.layers = layers

        self.user_emb = nn.Embedding(self.num_users, self.emb_size)
        self.item_emb = nn.Embedding(self.num_items, self.emb_size)

        self.linear = nn.ModuleList()
        for i in range(1, len(layers)):
            self.linear.append(nn.Linear(layers[i-1], layers[i]))
        
        self.final = nn.Sequential(
            nn.Linear(layers[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, users, items):
        user_emb = self.user_emb(users)
        item_emb = self.item_emb(items)
        user_item = torch.cat([user_emb, item_emb], dim=-1)
        linear_tensor = user_item
        for i in range(len(self.linear)):
            linear_tensor = self.linear[i](linear_tensor)
        output = self.final(linear_tensor)
        return output

