'''
Tweet2Vec encoder
'''

import numpy as np
import lasagne
import theano
import theano.tensor as T
import random
import sys
import time
import cPickle as pkl
import copy

from collections import OrderedDict, defaultdict
from settings import MAX_LENGTH, N_CHAR, WDIM
from model import tweet2vec, load_params

class Batch():
    
    def __init__(self, data, batch_size=512, maxlen=MAX_LENGTH):
        self.batch_size = batch_size
        self.data = data
        self.maxlen = maxlen

        self.prepare()
        self.reset()

    def prepare(self):
        # lengths
        self.lengths = [len(list(cc)) for cc in self.data]
        self.length_dict = defaultdict(list)
        for i,l in enumerate(self.lengths):
            self.length_dict[l].append(i)
        self.len_unique = self.length_dict.keys()

        # remove any overly long sentences
        if self.maxlen:
            self.len_unique = [ll for ll in self.len_unique if ll <= self.maxlen]

        self.len_counts = [len(self.length_dict[ii]) for ii in self.len_unique]
        self.len_curr_counts = copy.copy(self.len_counts)
        self.curr_len_pos = 0
        self.index_pos = 0

    def reset(self):
        self.len_curr_counts = copy.copy(self.len_counts)
        self.curr_len_pos = 0
        self.index_pos = 0

    def next(self):
        if self.len_curr_counts[self.curr_len_pos] <= 0:
            self.curr_len_pos += 1
            self.index_pos = 0
        if self.curr_len_pos >= len(self.len_unique):
            raise StopIteration

        # current batch
        current_batch_size = np.minimum(self.batch_size, self.len_curr_counts[self.curr_len_pos])

        # current indices
        current_indices = self.length_dict[self.len_unique[self.curr_len_pos]][self.index_pos:self.index_pos+current_batch_size]
        self.index_pos += current_batch_size
        self.len_curr_counts[self.curr_len_pos] -= current_batch_size

        # return data and indices
        batch = [self.data[ii] for ii in current_indices]

        return batch, current_indices

    def __iter__(self):
        return self

def prepare_data(seqs_x, chardict, n_chars=N_CHAR):
    """
    Put the data into format useable by the model
    """
    # remove never before seen characters and convert to index
    seqsX = []
    for cc in seqs_x:
        ccf = [c for c in list(cc) if c in chardict]
        seqsX.append([chardict[c] if chardict[c] < n_chars else 0 for c in ccf])
    seqs_x = seqsX

    lengths_x = [len(s) for s in seqs_x]

    n_samples = len(seqs_x)

    x = np.zeros((n_samples,MAX_LENGTH)).astype('int32')
    x_mask = np.zeros((n_samples,MAX_LENGTH)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[idx,:lengths_x[idx]] = s_x
        x_mask[idx,:lengths_x[idx]] = 1.

    return np.expand_dims(x,axis=2), x_mask

def main(data_path, model_path, save_path):

    print("Preparing Data...")

    # Load data and dictionary
    with open(data_path,'r') as f:
        X = f.read().splitlines()
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    n_char = len(chardict.keys()) + 1

    # Prepare data for encoding
    batches = Batch(X)

    # Load model
    print("Loading model params...")
    params = load_params('%s/model.npz' % model_path)

    # Build encoder
    print("Building encoder...")

    # Theano variables
    tweet = T.itensor3()
    t_mask = T.fmatrix()

    # Embeddings
    emb_t = tweet2vec(tweet, t_mask, params, n_char)[0]

    # Theano function
    f_enc = theano.function([tweet, t_mask], emb_t)

    # Encode
    print("Encoding data...")
    features = np.zeros((len(X),WDIM), dtype='float32')
    it = 0
    for x,i in batches:
        if it % 100 == 0:
            print("Minibatch {}".format(it))
        it += 1

        xp, x_mask = prepare_data(x, chardict)
        ff = f_enc(xp, x_mask)
        for ind, idx in enumerate(i):
            features[idx] = ff[ind]

    # Save
    with open(save_path, 'w') as o:
        np.save(o, features)

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])
