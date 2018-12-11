import interactions
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp


class MusicInteraction(interactions.Interactions):
    '''
    This class inherits Interactions and modify some methods from it
    to fit different data format from music data.
    
    The music data have following format:
    (seqid, user, item, item, item, ...)
    
    Each line represent a  music play history subsequence.
    
    Note that user and item should be encoded as postive interger.
    
    Parameters
    ----------
    file_path: file contains (seqid, user, item,[item,...] ) triplets
    user_map: dict of user mapping
    item_map: dict of item mapping
    '''
    def __init__(self,file_path,
                 num_items=None):        
        item_set = set()
        user_ids = list()        
        seq_items = OrderedDict()
        seqid_userid = dict()

        with open(file_path,'r') as f:
            print("...")
            print(f"Now loading data from {file_path}")
            for line in tqdm(f):
                seqid,uid,*items = line.strip().split(",")                        
                uid = int(uid)
                seqid = int(seqid)
                items = [int(item) for item in items]

                user_ids.append(uid)
                for i in items:
                    item_set.add(i)
                seq_items[seqid] = items
                seqid_userid[seqid] = uid

        if not num_items:
            #num_items = len(item_set) + 1
            num_items = max(item_set) +1 
        #seq_ids = np.array(list(seq_items.keys()))
        #seq_items = np.array(np.array(s) for s in seq_items)
        #assert len(user_ids) == len(seq_ids), "length of user_ids, seq_ids not the same!"        

        self.num_users = len(set(user_ids))
        self.num_items = num_items
        self.num_seqs = len(seq_items)        
        self.user_ids = np.array(user_ids)
        self.seq_items = seq_items # OrderDict
        self.seqid_userid = seqid_userid
        #self.item_set = item_set
        #self.seq_ids = seq_ids

        #self.user_map = user_map
        #self.item_map = item_map

        self.sequences = None
        self.test_sequences = None
            
        #print("num_users :{}\nnum_items:{}\nnum_seqs:{}".format(self.num_users,self.num_items,self.num_seqs))

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix, in which row represent seqs and col represent items.        
        """        
        d = []
        row = []
        col = []
        for k,v in self.seq_items.items():
            row = row + [k]*len(v)
            col = col + v
            d = d + [1]*len(v)
        print("row:{}\ncol:{}\nd:{}\nnum_seqs:{}\nnum_items:{}".format(max(row),max(col),len(d),self.num_seqs,self.num_items))
        coo = sp.coo_matrix((d,(row,col)),
                             shape =(self.num_seqs,self.num_items)) #(row,col)處為1，其餘為0  

        return coo


    
    def to_sequence(self, sequence_length=5, target_length=1):
        """
        Transform to sequence form.

        Valid subsequences of users' interactions are returned. For
        example, if a sub-sequences interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3 (should be 2 by hyades)
        will be be given by:

        sequences:

           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]

        targets:

           [[6, 7],
            [7, 8],
            [8, 9]]

        sequence for test (the last 'sequence_length' items of each user's sequence):

        [[5, 6, 7, 8, 9]]

        Parameters
        ----------

        sequence_length: int
            Sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        target_length: int
            Sequence target length.
        """

        max_sequence_length = sequence_length + target_length
        counts = [len(v) for v in self.seq_items.values()]
        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])
        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_targets = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)

        sequence_seqid = np.empty(num_subsequences,
                                  dtype=np.int64)

        test_sequences = np.zeros((self.num_seqs, sequence_length),
                                  dtype=np.int64)
        test_users = np.empty(self.num_seqs,
                              dtype=np.int64)
        test_seqid = np.empty(self.num_seqs,
                              dtype=np.int64)
    
        itered_seqid = set()
        for no,(seqid,uid,seq) in enumerate(self._get_seqnences(max_sequence_length)):            
            if seqid not in itered_seqid:
                test_sequences[seqid][:] = seq[-sequence_length:]
                test_users[seqid] = uid
                test_seqid[seqid] = seqid
                itered_seqid.add(seqid)
            sequences_targets[no][:] = seq[-target_length:]
            sequences[no][:] = seq[:sequence_length]
            sequence_users[no] = uid
            sequence_seqid[no] = seqid

        self.sequences = SequenceInteractions(sequence_users, sequence_seqid, sequences, sequences_targets)
        self.test_sequences = SequenceInteractions(test_users, test_seqid, test_sequences)

    def _get_seqnences(self,length):
        '''
            Iter over seq_items, and split seq to subseqs. 
            Yield (uid,seqid,seq) tuples.
        '''
        for uid,(seqid,seq) in zip(self.user_ids,
                                   self.seq_items.items()):            
            if len(seq)-length >= 0:
                # iter over seq to get subseq
                for i in range(len(seq),0,-1):
                    if i-length >= 0:
                        yield (seqid,uid,seq[i-length:i])
                    else:
                        break
            else:
                yield (seqid,uid,seq)


class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix.

    Parameters
    ----------
    user_ids: np.array
        sequence users
    seq_ids : np.array
        id from original sequence_id
    sequences: np.array
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    targets: np.array
        sequence targets
    """

    def __init__(self,
                 user_ids,
                 seq_ids,
                 sequences,
                 targets=None):
        self.user_ids = user_ids
        self.seq_ids = seq_ids
        self.sequences = sequences
        self.targets = targets

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]



        