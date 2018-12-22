import argparse

def examination(train_outfile,test_outfile):
	'''Check the overlaping rate : how many testing items doesn't appear in trainset.'''
	train_item = set()
	test_item = set()
	user_set = set()
	train_subseq_count = 0
	test_subseq_count = 0
	n_user = 0
	n_item = 0

	print("now loading train data...")
	with open(train_outfile,'rt') as train:
		for no,l in enumerate(train):
			_,*items = l.strip().split(",")
			for i in items:
				train_item.add(int(i))
	train_subseq_count = no+1

	print("now loading test data...\n\n\n")
	with open(test_outfile,'rt') as test:
		for no,l in enumerate(test):
			user,*items = l.strip().split(",")
			user_set.add(user)
			for i in items:
				test_item.add(int(i))
	test_subseq_count = no+1
	
	n_user = len(user_set)
	n_item = len(train_item.union(test_item))
	crossover_count = len(test_item.intersection(train_item)) 

	print("****   SUMMARY   ****")
	print("n_user : {}, n_item: {}".format(n_user,n_item))
	print("* Train: ")
	print("    sub_seqs: {:>8}, items:{:>8}".format(train_subseq_count,len(train_item)))
	print("* Test: ")
	print("    sub_seqs: {:>8}, items:{:>8}".format(test_subseq_count,len(test_item)))
	print("    overlap with train: {}, {}".format(crossover_count,crossover_count/len(test_item)))
	print("*********")

	return (n_user,n_item)
			                                                                   
			


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_outfile', type=str, default='data/train')
	parser.add_argument('--test_outfile', type=str, default='data/test')    
	config = parser.parse_args()

	examination(config.train_outfile,
		        config.test_outfile)
