import numpy as np

import torch
import torch.nn.functional as F
import random

activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': F.tanh, 'sigm': torch.sigmoid}

def gpu(tensor, gpu=False):

    if gpu:
        return tensor.cuda()
    else:
        return tensor


def cpu(tensor):

    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def str2bool(v):
    return v.lower() in ('true')



#hyades
def examination(train_outfile,test_outfile):
    '''
        Check the overlaping rate : how many testing items doesn't appear in trainset.
        Also return n_user,n_item,item_map for model use
    '''
    train_item = set()
    test_item = set()
    user_set = set()
    item_map = dict()
    item_map_count=0
    train_subseq_count = 0
    test_subseq_count = 0
    n_user = 0
    n_item = 0

    with open(train_outfile,'rt') as train:
        for no,l in enumerate(train):
            _,*items = l.strip().split(",")
            for i in items:
                i = int(i)
                if i not in item_map:
                    item_map_count +=1
                    item_map[i] = [item_map_count,1]
                else:
                    item_map[i][1] += 1

                train_item.add(int(i))
    train_subseq_count = no+1

    with open(test_outfile,'rt') as test:
        for no,l in enumerate(test):
            user,*items = l.strip().split(",")
            user_set.add(int(user))
            for i in items:
                test_item.add(int(i))
    test_subseq_count = no+1
    
    n_user = max(user_set) +1
    n_item = max(train_item.union(test_item))
    crossover_count = len(test_item.intersection(train_item)) 

    print("****   SUMMARY   ****")
    print("n_user : {}, n_item: {}".format(n_user,n_item))
    print("* Train: ")
    print("    sub_seqs: {:>8}, items:{:>8}".format(train_subseq_count,len(train_item)))
    print("* Test: ")
    print("    sub_seqs: {:>8}, items:{:>8}".format(test_subseq_count,len(test_item)))
    print("    overlap with train: {}, {}".format(crossover_count,crossover_count/len(test_item)))
    print("*********")

    return (n_user,n_item,item_map)    


#
def precision_recall(pred,target,at=[1,5,10]):
    prec_res = list()
    recall_res = list()
    target = set(target)
    n = len(target)
    for l in at:
        cur_pred = set(pred[0:l])
        prec = cur_pred.intersection(target)
        recall = target.intersection(cur_pred)
        prec_res.append(len(prec)/len(cur_pred))
        recall_res.append(len(recall)/n)
    return (prec_res,recall_res)


