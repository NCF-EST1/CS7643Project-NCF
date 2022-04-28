import torch
import torch.nn as nn

class GMF(nn.Module):

    def __init__(self, num_users, num_items, emb_size):
        super(GMF, self).__init__()
        self.name = "GMF"
        self.num_users = num_users
        self.num_items = num_items
        self.emb_size = emb_size

        self.user_emb = nn.Embedding(self.num_users, self.emb_size)
        self.item_emb = nn.Embedding(self.num_items, self.emb_size)

        self.linear = nn.Sequential(
            nn.Linear(self.emb_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, users, items):
        user_emb = self.user_emb(users)
        item_emb = self.item_emb(items)
        user_item = torch.mul(user_emb, item_emb)
        output = self.linear(user_item)
        return output

