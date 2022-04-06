import torch
import torch.nn as nn

class NCF(nn.Module):

    def __init__(self, num_users, num_items, emb_size):
        super(NCF, self).__init__()
        self.name = "NCF"
        self.num_users = num_users
        self.num_items = num_items
        self.emb_size = emb_size

        self.user_emb = nn.Embedding(self.num_users, self.emb_size)
        self.item_emb = nn.Embedding(self.num_items, self.emb_size)

        self.linear = nn.Sequential(
            nn.Linear(self.emb_size * 2, 64),
            # nn.Linear(64, 32),
            # nn.Linear(32, 16),
            # nn.Linear(16, 8),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, users, items):
        user_emb = self.user_emb(users)
        item_emb = self.item_emb(items)
        user_item = torch.cat([user_emb, item_emb], dim=-1)
        output = self.linear(user_item)
        return output

