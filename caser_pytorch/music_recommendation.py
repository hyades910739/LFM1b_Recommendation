import argparse
import os
from time import time
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from caser import Caser
from evaluation import evaluate_ranking
#from interactions import Interactions
from utils import *
#from music_interactions import MusicInteraction,SequenceInteractions
from train_caser import Recommender

class Music_Recommender(Recommender):
    def __init__(self,
                 train_path,
                 test_path,
                 item_path,
                 train_length,
                 test_length,
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
        self.train_length = train_length
        self.test_length = test_length
        if item_path:
            self.item_path = item_path
            self.item_cumsum = self.get_item_cumsum()
                
    @property
    def train_path(self):
        return self._train_path
    
    @train_path.setter
    def train_path(self,value):
        if not os.path.isfile(value):
            raise FileNotFoundError("No file found for train_path : {}".format(value))
        else:
            self._train_path = value

    @property
    def test_path(self):
        return self._test_path
    
    @test_path.setter
    def test_path(self,value):
        if not os.path.isfile(value):
            raise FileNotFoundError("No file found for test_path : {}".format(value))
        else:
            self._test_path = value

    @property
    def item_path(self):
        return self._item_path
    
    @item_path.setter
    def item_path(self,value):
        if not os.path.isfile(value):
            raise FileNotFoundError("No file found for item_path : {}".format(value))
        else:
            self._item_path = value        
    

    def get_item_cumsum(self):
        '''read item counts and calculate item_count cusums for  negative sampling'''
        with open(self.item_path,'rt') as f:
            counts = list()
            for line in f:
                _,_,count = line.strip().split(",")

                counts.append(int(count))
        return np.cumsum(counts)

    def minibatch_io(self,random=True):
        '''
            read data by iterator

            return: tuples of np.arrays : (uids,trainseqs,testseqs) 
            with dimensions:
                uids : (batch_size)
                trainseqs : (batch_size, train_length)
                testseqs : (batch_size, test_length)
        '''
        length = self.train_length + self.test_length
        train_length = self.train_length
        test_length = self.test_length
        max_storage =  self._batch_size*100
        batch_size = self._batch_size
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
                    cur_trainseqs.append(seq[0:train_length])
                    cur_testseqs.append(seq[-test_length:])
                    cur_uid.append(uid)
                    count +=1

                    if count == max_storage:
                        cur_uid = np.array(cur_uid)
                        cur_trainseqs = np.array(cur_trainseqs)
                        cur_testseqs = np.array(cur_testseqs)
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
                cur_uid = np.array(cur_uid)
                cur_trainseqs = np.array(cur_trainseqs)
                cur_testseqs = np.array(cur_testseqs)                
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

    def _initialize(self):
        n_user,n_item = examination(self.train_path,self.test_path)

        self._num_items = n_item +1 # for 0 padding
        self._num_users = n_user
        print(self._num_items)
        print(self._num_users)
        
        self._net = gpu(Caser(self._num_users,
                              self._num_items,
                              self.model_args), self._use_cuda)

        self._optimizer = optim.Adam(
            self._net.parameters(),
            weight_decay=self._l2,
            lr=self._learning_rate)                        

    def fit(self, verbose=False):
        """
        The general training loop to fit the model

        Parameters
        ----------

        train: :class:`spotlight.interactions.Interactions`
            training instances, also contains test sequences
        test: :class:`spotlight.interactions.Interactions`
            only contains targets for test sequences
        verbose: bool, optional
            print the logs
        """

        if not self._initialized:
            self._initialize()

        start_epoch = 0

        for epoch_num in range(start_epoch, self._n_iter):

            t1 = time()
            # set model to training model
            self._net.train()
            epoch_loss = 0.0

            for minibatch_num,(cur_uid,cur_trainseqs,cur_testseqs) in enumerate(self.minibatch_io(random=True)):
                
                negative_samples = self._generate_negative_samples(np.concatenate((cur_trainseqs,cur_testseqs),axis=1),
                                                                   self._neg_samples)

                # shape: (# of subseqs, sequence_length)
                sequences_tensor = gpu(torch.from_numpy(cur_trainseqs),
                                       self._use_cuda)
                # shape: (# of subseqs, 1)
                user_tensor = gpu(torch.from_numpy(cur_uid),
                                  self._use_cuda)
                user_tensor = user_tensor.unsqueeze_(1)
                # shape: (# of subseqs, targets_length)
                item_target_tensor = gpu(torch.from_numpy(cur_testseqs),
                                         self._use_cuda)
                # shape: (# of subseqs, self._neg_samples * T)
                item_negative_tensor = gpu(torch.from_numpy(negative_samples),
                                           self._use_cuda)

                sequence_var = Variable(sequences_tensor)
                user_var = Variable(user_tensor)
                item_target_var = Variable(item_target_tensor)
                item_negative_var = Variable(item_negative_tensor)


                target_prediction = self._net(sequence_var,
                                              user_var,
                                              item_target_var)
                negative_prediction = self._net(sequence_var,
                                                user_var,
                                                item_negative_var,
                                                use_cache=True)

                self._optimizer.zero_grad()
                # compute the binary cross-entropy loss
                positive_loss = -torch.mean(torch.log(F.sigmoid(target_prediction)))
                negative_loss = -torch.mean(torch.log(1 - F.sigmoid(negative_prediction)))
                loss = positive_loss + negative_loss

                epoch_loss += loss.data[0]

                loss.backward()
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1

            t2 = time()
            if verbose and (epoch_num + 1) % 1 == 0:
                precision, recall, mean_aps = evaluate_ranking(self, test, train, k=[1, 5, 10])
                output_str = "Epoch %d [%.1f s]\tloss=%.4f, map=%.4f, " \
                             "prec@1=%.4f, prec@5=%.4f, prec@10=%.4f, " \
                             "recall@1=%.4f, recall@5=%.4f, recall@10=%.4f, [%.1f s]" % (epoch_num + 1,
                                                                                         t2 - t1,
                                                                                         epoch_loss,
                                                                                         mean_aps,
                                                                                         np.mean(precision[0]),
                                                                                         np.mean(precision[1]),
                                                                                         np.mean(precision[2]),
                                                                                         np.mean(recall[0]),
                                                                                         np.mean(recall[1]),
                                                                                         np.mean(recall[2]),
                                                                                         time() - t2)
                print(output_str)
            else:
                output_str = "Epoch %d [%.1f s]\tloss=%.4f [%.1f s]" % (epoch_num + 1,
                                                                        t2 - t1,
                                                                        epoch_loss,
                                                                        time() - t2)
                print(output_str)
    

    def _generate_negative_samples(self,seqs,n):
        n_seqs = seqs.shape[0]
        # generate negative samples, item counts as sampling weight 
        
        negative_samples_flat = self.item_cumsum.searchsorted(np.random.uniform(0,self.item_cumsum[-1],n_seqs*n))
        
        negative_samples = negative_samples_flat.reshape((n_seqs,n))
        # negative samples should not be in seqs(trainseq and testseqs)
        # if so, replace it randomly
        for row in range(n_seqs):
            for col in range(n):
                if negative_samples[row,col] in seqs[row,]:
                    match = True
                    while match:
                        sel = np.random.choice(negative_samples_flat)
                        if sel not in seqs[row,]:
                            negative_samples[row,col] = sel
                            match = False

        return negative_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_path', type=str, default='/home/hyades/Documents/py/LFM-1b/data/train')
    parser.add_argument('--test_path', type=str, default='/home/hyades/Documents/py/LFM-1b/data/test')
    parser.add_argument('--item_path', type=str, default='/home/hyades/Documents/py/LFM-1b/data/itemid_map')

    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)

    config = parser.parse_args()

    # model dependent arguments
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')

    model_config = model_parser.parse_args()
    model_config.L = config.L

    # set seed
    set_seed(config.seed,
             cuda=config.use_cuda)

    model = Music_Recommender(train_path=config.train_path,
                              test_path=config.test_path,
                              item_path=config.item_path,
                              train_length=config.L,
                              test_length=config.T,
                              n_iter=config.n_iter,
                             batch_size=config.batch_size,
                             learning_rate=config.learning_rate,
                             l2=config.l2,
                             neg_samples=config.neg_samples,
                             model_args=model_config,
                             use_cuda=config.use_cuda)

    model.fit(verbose=False)



