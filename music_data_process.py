import os
import argparse
from itertools import islice
from tqdm import tqdm
import numpy as np
from music_data_examination import examination


def data_io(in_file,
            train_outfile,
            test_outfile,
            userid_map_outfile,
            itemid_map_outfile,
            n_user=5000,
            timespace=(60,3600),
            seqlen=(20,30),
            iter_limit=None,
            n_test=3,
            min_subseq=50):
    '''
    process music data,split the sequence to subsequences, and save to other files.

    
    Parameter:
    ------------------
    in_file: str
        input file location  
    train_outfile: str
        the train_output file location and filename.     
    test_outfile: str
        the test_output file location and filename.
    userid_map_outfile: str
        The userid mapping file location and filename.
        In train_test file, the userid is encode as 0,1,2,... 
        This file preserve the mapping from this encode to original userid.
    itemid_map_outfile: str
        The itemid mapping file location and filename.
        In train_test file, the itemid is encode as 1,2,3,...
        This file preserve the mapping from this encode to original itemid.
    n_user: int
        how many user should we include in train/test data
        timespace: tuple(int,int)
    the cutoff point to split LE sequences to sub-sequences.
        MUST be (lower,higher) format.
    seqlen: tuple(int,int)
        the minimum and maximum sub-sequence length to preserve in train/test set.
        MUST be (lower,higher) format.

    iter_limit: int or None
        how many LEs should we include in train/test
        default is None, then we dont stop iteration due to iter_limit.
    n_test: int
        how many sub-sequences should be include in test set
        if user a have 300 sub-seq and n_test=5
        then we'll RANDOMLY select 5 sub-seqs into test set, and others remain in train set

    min_subseq: int
        how many sub-seqs should one user have to include in train/test set
        if lower, the sub-seqs of current user will not include in train/test set   

    '''

    if not os.path.isfile(in_file):
        raise FileNotFoundError("in_file not found!")

    with open(in_file,'rt') as f:
        n_user_count = 0
        n_iter = 0
        uid_map = 0
        iid_map = dict()
        iid_no = 1
        prev_user = None
        item_list = list()
        for line in islice(f,0,iter_limit):
            uid,_,_,iid,time = line.strip().split("\t")

            uid = int(uid)
            iid = int(iid)
            time = int(time)
            n_iter += 1
            if iter_limit is not None:
                if n_iter > iter_limit:
                    break
            if iid not in iid_map:
                iid_map[iid] = iid_no
                iid_no += 1
            if prev_user != uid:
                if prev_user is None:
                    prev_user = uid
                else:
                    res = subseq_process(item_list,prev_user,train_outfile,
                                         test_outfile,userid_map_outfile,timespace,
                                         seqlen,n_test,min_subseq,
                                         uid_map,iid_map)
                    item_list = list()
                    prev_user = uid
                    if res : 
                        n_user_count += 1
                        uid_map += 1
                    if n_user_count >= n_user:
                        break           
            item_list.append((iid,time))
        #finally:
        item_map_output(itemid_map_outfile,iid_map)

    print("\n****")
    print(" {} LEs, {} users processed.".format(n_iter-1,n_user_count-1))
    print("****")
    print("\nnow examining...")
    examination(train_outfile,test_outfile)



def item_map_output(itemid_map_outfile,
                    iid_map):
    with open(itemid_map_outfile,'a+') as f:
        for k,v in iid_map.items():
            l = str(v) + "," + str(k) + "\n"
            f.writelines(l) 

    
            
def subseq_process(item_list,
                   uid,
                   train_outfile,
                   test_outfile,
                   userid_map_outfile,
                   timespace,
                   seqlen,
                   n_test,
                   min_subseq,
                   uid_map,
                   iid_map):
    '''main function to conduct sub-seq split/filter'''
    
    MINTIME,MAXTIME = timespace
    MINLEN,MAXLEN = seqlen
    res = list()
    #sort list by time:
    item_list = sorted(item_list,key= lambda x:x[1])
    t = np.diff([i[1] for i in item_list])
    item_list = [iid_map[i[0]] for i in item_list]

    # find cutoff point
    sel = (t>MAXTIME) | (t<MINTIME)
    cutoff = np.where(sel)[0]
    past = None
    for i in cutoff:
        if not past:
            cur = item_list[0:(i+1)]
            cur = remove_repeat(cur)
            if(len(cur)>=MINLEN and len(cur)<=MAXLEN): 
                cur = [uid_map] + cur
                cur = [str(i) for i in cur]
                cur = ",".join(cur) + "\n"
                res.append(cur)             
            past = i+1
        else:
            cur = item_list[past:(i+1)]
            cur = remove_repeat(cur)
            if(len(cur)>=MINLEN and len(cur)<=MAXLEN): 
                cur = [uid_map] + cur
                cur = [str(i) for i in cur]             
                cur = ",".join(cur) + "\n"
                res.append(cur)
            past = i+1

    cur = item_list[past:]
    cur = remove_repeat(cur)
    if(len(cur)>=MINLEN and len(cur)<=MAXLEN):
        cur = [uid_map] + cur
        cur = [str(i) for i in cur]
        cur = ",".join(cur) + "\n"
        res.append(cur)     
    if len(res)>min_subseq:
        subseq_output(res,train_outfile,test_outfile,userid_map_outfile,n_test,uid,uid_map)
        return True
    else:
        return False

    


def remove_repeat(seq):
    '''
    remove continuous LEs, keep only one.
    for example : LE = [1,3,3,3,4,5,5] will become [1,3,4,5]
    '''
    #item = (i[0] for i in seq)
    index = None
    sel = list()
    for no,i in enumerate(seq):
        if not index: 
            index = i
            sel.append(no)
            continue
        if i==index:
            pass
        else:
            index = i
            sel.append(no)
    return [seq[i] for i in sel]        

def subseq_output(res,
                  train_outfile,
                  test_outfile,
                  userid_map_outfile,
                  n_test,
                  uid,
                  uid_map):
    '''
       write subseq to train_file and test_file
       n randomly selected sub-sequence will write to testfile
       others goes to trainfile
    '''
    sel = np.random.choice(range(len(res)),n_test,replace=False)

    with open(userid_map_outfile,"a+") as map_:
        l = str(uid_map) + "," + str(uid) + "\n"
        map_.writelines(l)


    with open(train_outfile,'a+') as train, \
         open(test_outfile,'a+') as test:        

        for no,line in enumerate(res):
            if no in sel:
                test.writelines(line)
            else:
                train.writelines(line)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default='data/LFM-1b_LEs.txt')
    parser.add_argument('--train_outfile', type=str, default='data/train')
    parser.add_argument('--test_outfile', type=str, default='data/test')    
    parser.add_argument('--userid_map_outfile', type=str, default='data/userid_map')    
    parser.add_argument('--itemid_map_outfile', type=str, default='data/itemid_map')    
    parser.add_argument('--n_user', type=int, default=3000)
    parser.add_argument('--mintime', type=int, default=60)
    parser.add_argument('--maxtime', type=int, default=3600)
    parser.add_argument('--minseqlen', type=int, default=20)
    parser.add_argument('--maxseqlen', type=int, default=30)
    parser.add_argument('--iter_limit', type=int, default=0)
    parser.add_argument('--n_test', type=int, default=3)
    parser.add_argument('--min_subseq', type=int, default=20)

    config = parser.parse_args()
    if config.iter_limit == 0 :
        config.iter_limit = None

    if os.path.isfile(config.train_outfile) or os.path.isfile(config.test_outfile) \
                                            or os.path.isfile(config.userid_map_outfile) \
                                            or os.path.isfile(config.itemid_map_outfile):
        print("***   WARNING : outfile exist!   ***")
        print("train_outfile : '{}'".format(config.train_outfile),end=",")
        print(" or test_outfile : '{}'".format(config.test_outfile))
        print(" or userid_map_outfile : '{}'".format(config.userid_map_outfile),end=",")
        print(" or itemid_map_outfile : '{}'".format(config.itemid_map_outfile))
        print("enter y/Y to overlap it, else exit.")
        overlap = input()
        if overlap !='y' and overlap !="Y":
            print("exit!")
            exit()
        if os.path.isfile(config.train_outfile) : os.remove(config.train_outfile)
        if os.path.isfile(config.test_outfile): os.remove(config.test_outfile)
        if os.path.isfile(config.userid_map_outfile): os.remove(config.userid_map_outfile)
        if os.path.isfile(config.itemid_map_outfile): os.remove(config.itemid_map_outfile)


    data_io(in_file = config.in_file,
            train_outfile = config.train_outfile,
            test_outfile = config.test_outfile,
            userid_map_outfile = config.userid_map_outfile,
            itemid_map_outfile = config.itemid_map_outfile,
            n_user= config.n_user,
            timespace=(config.mintime,config.maxtime),
            seqlen=(config.minseqlen,config.maxseqlen),
            iter_limit=config.iter_limit,
            n_test=config.n_test,
            min_subseq=config.min_subseq)
