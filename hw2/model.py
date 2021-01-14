import torch
import torch.nn as nn
import torch.nn.functional as F

class BPR_MF(nn.Module):
    def __init__(self, user_dim, item_dim, hidden_dim=512):
        super(BPR_MF, self).__init__()
        self.user_embedding = nn.Embedding(user_dim, hidden_dim)
        self.item_embedding = nn.Embedding(item_dim, hidden_dim)

        nn.init.orthogonal_(self.user_embedding.weight)
        nn.init.orthogonal_(self.item_embedding.weight)



    def forward(self, user, item, neg=None, val=False):
        user_latent = self.user_embedding(user)
        item_latent = self.item_embedding(item)
        if val == False:
            score_1 = torch.bmm(user_latent.unsqueeze(1), item_latent.unsqueeze(2)).squeeze(2).squeeze(1)
        else:
            score_1 = torch.bmm(user_latent.unsqueeze(1), item_latent.permute(0, 2, 1))
            return torch.sigmoid(score_1)
        #return torch.sigmoid(score_1)

        #return score_1
        #if neg != None:

        item_latent_2 = self.item_embedding(neg)


        score_2 = torch.bmm(user_latent.unsqueeze(1), item_latent_2.unsqueeze(2)).squeeze(2).squeeze(1)
        log_prob = F.logsigmoid(score_1-score_2).sum()

        return -log_prob

class BCE_MF(nn.Module):
    def __init__(self, user_dim, item_dim, hidden_dim=64):
        super(BCE_MF, self).__init__()
        self.user_embedding = nn.Embedding(user_dim, hidden_dim)
        self.item_embedding = nn.Embedding(item_dim, hidden_dim)

        nn.init.orthogonal_(self.user_embedding.weight)
        nn.init.orthogonal_(self.item_embedding.weight)

        self.loss_function = nn.BCELoss()


    def forward(self, user, item, neg=None, val=False):
        user_latent = self.user_embedding(user)
        item_latent = self.item_embedding(item)
        if val == False:
            score_1 = torch.bmm(user_latent.unsqueeze(1), item_latent.unsqueeze(2)).squeeze(2).squeeze(1)
        else:
            score_1 = torch.bmm(user_latent.unsqueeze(1), item_latent.permute(0, 2, 1))
            return torch.sigmoid(score_1)
        #return torch.sigmoid(score_1)

        #return score_1
        #if neg != None:

        item_latent_2 = self.item_embedding(neg)
        score_2 = torch.bmm(user_latent.unsqueeze(1), item_latent_2.unsqueeze(2)).squeeze(2).squeeze(1)

        #log_prob = F.logsigmoid(score_1-score_2).sum()
        output = torch.sigmoid(torch.cat([score_1, score_2], dim=0))
        target = torch.cat([torch.ones(score_1.size()[0]), torch.zeros(score_1.size()[0])], dim=0).cuda()
        loss = self.loss_function(output, target)

        return loss
