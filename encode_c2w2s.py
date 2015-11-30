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
import io

from collections import OrderedDict, defaultdict
from settings import MAX_LENGTH, MAX_WORD_LENGTH, MAX_SEQ_LENGTH, N_CHAR, SDIM
from model import char2word2vec, load_params

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

def prepare_data_c2w2s(seqs_x, chardict, maxwordlen=MAX_WORD_LENGTH, maxseqlen=MAX_SEQ_LENGTH, n_chars=N_CHAR):
    """
    Put the data into format useable by the model
    """
    n_samples = len(seqs_x)
    x = numpy.zeros((n_samples,MAX_SEQ_LENGTH,MAX_WORD_LENGTH)).astype('int32')
    x_mask = numpy.zeros((n_samples,MAX_SEQ_LENGTH,MAX_WORD_LENGTH)).astype('float32')

    # Split words and replace by indices
    for seq_id, cc in enumerate(seqs_x):
        words = cc.split()
        for word_id, word in enumerate(words):
            if word_id >= MAX_SEQ_LENGTH:
                break
            c_len = min(MAX_WORD_LENGTH, len(word))
            x[seq_id,word_id,:c_len] = [chardict[c] if c in chardict and chardict[c] < n_chars else 0 for c in list(word)[:c_len]]
            x_mask[seq_id,word_id,:c_len] = 1.

    return numpy.expand_dims(x,axis=3), x_mask

def main(data_path, model_path, save_path):

    print("Preparing Data...")

    # Load data and dictionary
    X = []
    with io.open(data_path,'r',encoding='utf-8') as f:
        for line in f:
            X.append(line.rstrip('\n'))
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
    tweet = T.itensor4()
    t_mask = T.ftensor3()

    # Embeddings
    emb_t = char2word2vec(tweet, t_mask, params, n_char)[0]

    # Theano function
    f_enc = theano.function([tweet, t_mask], emb_t)

    # Encode
    print("Encoding data...")
    print("Input data {} samples".format(len(X)))
    features = np.zeros((len(X),SDIM), dtype='float32')
    it = 0
    for x,i in batches:
        if it % 100 == 0:
            print("Minibatch {}".format(it))
        it += 1

        xp, x_mask = prepare_data_c2w2s(x, chardict)
        ff = f_enc(xp, x_mask)
        for ind, idx in enumerate(i):
            features[idx] = ff[ind]

    # Save
    with open(save_path, 'w') as o:
        np.save(o, features)

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])
