import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import activation_getter
import numpy as np

class Caser(nn.Module):
    """
    Convolutional Sequence Embedding Recommendation Model (Caser)[1].

    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18

    Parameters
    ----------

    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, num_users, num_items, model_args,pre_train=None):
        super(Caser, self).__init__()
        self.args = model_args

        # init args
        L = self.args.L
        dims = self.args.d
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # user and item embeddings
        if isinstance(pre_train,np.ndarray):
            pre_train = torch.from_numpy(pre_train)
            self.item_embeddings = nn.Embedding.from_pretrained(pre_train)
            dims = pre_train.shape[1]
            self.item_embeddings.requires_grad=False
        else:            
            self.item_embeddings = nn.Embedding(num_items, dims)
        self.user_embeddings = nn.Embedding(num_users, dims)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, dims+dims)
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        #self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        #batch norm
        self.bn2 = nn.BatchNorm1d(dims*2)

        self.cache_x = None

    def forward(self, seq_var, user_var, item_var, use_cache=False, for_pred=False):
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets). Note that we can cache 'x' to
        save computation for negative predictions. Because when computing
        negatives, the (user, sequence) are the same, thus 'x' will be the
        same as well.

        Parameters
        ----------

        seq_var: torch.autograd.Variable
            a batch of sequence
        user_var: torch.autograd.Variable
            a batch of user
        item_var: torch.autograd.Variable
            a batch of items
        use_cache: boolean, optional
            Use cache of x. Set to True when computing negatives.
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        if not use_cache:
            # Embedding Look-up
            item_embs = self.item_embeddings(seq_var).unsqueeze(1).float()  # use unsqueeze() to get 4-D
            user_emb = self.user_embeddings(user_var).squeeze(1).float( )

            # Convolutional Layers
            out, out_h, out_v = None, None, None
            # vertical conv layer
            if self.n_v:
                out_v = self.conv_v(item_embs).squeeze(2)
                out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

            # horizontal conv layer
            out_hs = list()
            if self.n_h:
                for conv in self.conv_h:
                    conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                    pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                    out_hs.append(pool_out)
                out_h = torch.cat(out_hs, 1)  # prepare for fully connect

            # Fully-connected Layers
            out = torch.cat([out_v, out_h], 1)
            # apply dropout
            out = self.dropout(out)

            # fully-connected layer
            z = self.ac_fc(self.fc1(out))
            x = torch.cat([z, user_emb], 1)
            #batch norm!
            x = self.bn2(x)


            self.cache_x = x

        else:
            x = self.cache_x

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)
        if not for_pred:
            results = []
            for i in range(item_var.size(1)):
                w2i = w2[:, i, :]
                b2i = b2[:, i, 0]
                result = (x * w2i).sum(1) + b2i #只對選取的商品做 NN output，這邊在做全連結層的內積，dim = item_var.shape[0]
                results.append(result) #shape : item_var.shape(1),item_var.shape(0)
            res = torch.stack(results, 1) #shape : item_var.shape(0),item_var.shape(1)
        else:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (x * w2).sum(1) + b2

        return res
