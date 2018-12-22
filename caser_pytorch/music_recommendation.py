import argparse
from time import time
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

from caser import Caser
from evaluation import evaluate_ranking
#from interactions import Interactions
from utils import *
from music_interactions import MusicInteraction,SequenceInteractions
from train_caser import Recommender

class Music_Recommender(Recommender):
    def __init__(self,
                 train_path,
                 test_path,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 neg_samples=None,
                 learning_rate=None,
                 use_cuda=False,
                 model_args=None):

        super().__init__(n_iter,batch_size,l2,neg_samples,
                         learning_rate,use_cuda,model_args)
        self.train_path = train_path
        self.test_path = test_path
        
        
    @property
    def train_path(self):
        pass

    def batch_io(self,
                 train_length=5,
                 test_length=3,
                 batch_size=512,               
                 random=True):
        '''
            read data by iterator
        '''
        length = train_length + test_length
        max_storage =  batch_size*100
        count = 0

        with open(self.train_path,'rt') as f:
            cur_trainseqs = list()
            cur_testseqs = list()
            cur_uid = list()
            for line in f:
                uid,*items = line.strip().split(",")
                uid = int(uid)
                items = [int(i) for i in items]
                for i in range(len(items)-length+1):
                    seq = items[i:(i+length)]
                    cur_trainseqs.append(seq[0:length])
                    cur_testseqs.append(seq[-test_length:])
                    cur_uid.append(uid)
                    count +=1

                    if count == max_storage:
                        for i in range(0,max_storage,batch_size):
                            yield (cur_uid[i:i+batch_size],
                                   cur_trainseqs[i:i+batch_size],
                                   cur_testseqs[i:i+batch_size])
                        #reset cur after yield data.
                        cur_trainseqs = list()
                        cur_testseqs = list()
                        cur_uid = list()

            # remaining data (perhaps less then batch_size):
            for i in range(0,len(cur_uid),batch_size):
                if (i+batch_size) >len(cur_uid):
                    yield (cur_uid[i:],
                           cur_trainseqs[i:],
                           cur_testseqs[i:])
                else:               
                    yield (cur_uid[i:i+batch_size],
                           cur_trainseqs[i:i+batch_size],
                           cur_testseqs[i:i+batch_size])
    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        examination()

        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        self._net = gpu(Caser(self._num_users,
                              self._num_items,
                              self.model_args), self._use_cuda)

        self._optimizer = optim.Adam(
            self._net.parameters(),
            weight_decay=self._l2,
            lr=self._learning_rate)                        

    
















