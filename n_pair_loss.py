import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NPairLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def select_pair(self, label):
        label_col = label.unsqueeze(1)
        label_row = label.unsqueeze(0)
        similar_matrix = (label_col == label_row).byte() # byte() cast True or False to 1 / 0

        # same to similar_matrix.fill_diagonal_(0), to be compatible with pytorch <= 1.6
        self_mask = torch.eye(len(label), dtype=torch.int16) ^ 1
        pair_matrix = similar_matrix * self_mask

        anchor_idx, positive_idx = torch.nonzero(pair_matrix, as_tuple=True)

        _, unique_idx = np.unique(label[anchor_idx].cpu().numpy(), return_index=True)

        anchor_idx = anchor_idx[unique_idx]
        positive_idx = positive_idx[unique_idx]
        return anchor_idx, positive_idx

    def DotProductSimilarity(self, emb1, emb2):
        return torch.matmul(emb1, emb2.T)

    def forward(self, embedding, label):
        # select (anchor, positive) pairs for each class
        anchor_idx, positive_idx= self.select_pair(label)

        if (len(anchor_idx) == 0):
            return torch.Tensor([0])[0]

        # l2 normalization is significant for contrastive learning
        embedding_norm = F.normalize(embedding, p=2)

        anchor_embedding = embedding_norm[anchor_idx]

        positive_embedding = embedding_norm[positive_idx]

        similarity_matrix = self.DotProductSimilarity(anchor_embedding, positive_embedding)

        target_label = torch.arange(start=0, end=len(anchor_idx))

        ce_loss = nn.CrossEntropyLoss()

        loss = ce_loss(similarity_matrix, target_label)

        return loss

if __name__ == '__main__':

    input = torch.Tensor(np.random.rand(8, 768))
    label = torch.Tensor([1,1,1,0,0,0,1,1])
    loss = NPairLoss()
    print(loss(input, label))


