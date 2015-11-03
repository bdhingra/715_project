import numpy
import copy
from collections import OrderedDict

class BatchedTweets():

    def __init__(self, data, validation_size=1000, batch_size=128, maxlen=None):
        self.batch_size = 128
        self.validation = (data[0][0:validation_size], data[1][0:validation_size], data[2][0:validation_size])
        self.data = (data[0][validation_size:], data[1][validation_size:], data[2][validation_size:])
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.prepare()
        self.reset()

    def validation_set(self):
        return self.validation

    def prepare(self):
        self.caps = self.data[0]
        self.feats = self.data[1]
        self.feats2 = self.data[2]

        # find the unique lengths
        self.lengths = [len(list(cc)) for cc in self.caps]
        self.len_unique = numpy.unique(self.lengths)
        # remove any overly long sentences
        if self.maxlen:
            self.len_unique = [ll for ll in self.len_unique if ll <= self.maxlen]

        # indices of unique lengths
        self.len_indices = dict()
        self.len_counts = dict()
        for ll in self.len_unique:
            self.len_indices[ll] = numpy.where(self.lengths == ll)[0]
            self.len_counts[ll] = len(self.len_indices[ll])

        # current counter
        self.len_curr_counts = copy.copy(self.len_counts)

    def reset(self):
        self.len_curr_counts = copy.copy(self.len_counts)
        self.len_unique = numpy.random.permutation(self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0
            self.len_indices[ll] = numpy.random.permutation(self.len_indices[ll])
        self.len_idx = -1

    def next(self):
        count = 0
        while True:
            self.len_idx = numpy.mod(self.len_idx+1, len(self.len_unique))
            if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break
        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration()

        # get the batch curr_batch_size
        curr_batch_size = numpy.minimum(self.batch_size, self.len_curr_counts[self.len_unique[self.len_idx]])
        curr_pos = self.len_indices_pos[self.len_unique[self.len_idx]]
        # get the indices for the current batch
        curr_indices = self.len_indices[self.len_unique[self.len_idx]][curr_pos:curr_pos+curr_batch_size]
        self.len_indices_pos[self.len_unique[self.len_idx]] += curr_batch_size
        self.len_curr_counts[self.len_unique[self.len_idx]] -= curr_batch_size

        # 'feats' corresponds to the after and before sentences
        caps = [self.caps[ii] for ii in curr_indices]
        feats = [self.feats[ii] for ii in curr_indices]
        feats2 = [self.feats2[ii] for ii in curr_indices]

        return caps, feats, feats2

    def __iter__(self):
        return self

def prepare_data(seqs_x, seqs_y, seqs_z, chardict, maxlen=None, n_chars=20000):
    """
    Put the data into format useable by the model
    """
    seqsX = []
    seqsY = []
    seqsZ = []
    for cc in seqs_x:
        seqsX.append([chardict[c] if chardict[c] < n_chars else 1 for c in list(cc)])
    for cc in seqs_y:
        seqsY.append([chardict[c] if chardict[c] < n_chars else 1 for c in list(cc)])
    for cc in seqs_z:
        seqsZ.append([chardict[c] if chardict[c] < n_chars else 1 for c in list(cc)])
    seqs_x = seqsX
    seqs_y = seqsY
    seqs_z = seqsZ

    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]
    lengths_z = [len(s) for s in seqs_z]

    if maxlen != None:
        new_seqs_x = []
        new_seqs_y = []
        new_seqs_z = []
        new_lengths_x = []
        new_lengths_y = []
        new_lengths_z = []
        for l_x, s_x, l_y, s_y, l_z, s_z in zip(lengths_x, seqs_x, lengths_y, seqs_y, lengths_z, seqs_z):
            if l_x < maxlen and l_y < maxlen and l_z < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
                new_seqs_z.append(s_z)
                new_lengths_z.append(l_z)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        lengths_z = new_lengths_z
        seqs_z = new_seqs_z

        if len(lengths_x) < 1 or len(lengths_y) < 1 or len(lengths_z) < 1:
            return None, None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1
    maxlen_z = numpy.max(lengths_z) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    z = numpy.zeros((maxlen_z, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    z_mask = numpy.zeros((maxlen_z, n_samples)).astype('float32')
    for idx, [s_x, s_y, s_z] in enumerate(zip(seqs_x,seqs_y,seqs_z)):
        x[:lengths_x[idx],idx] = s_x
        x_mask[:lengths_x[idx]+1,idx] = 1.
        y[:lengths_y[idx],idx] = s_y
        y_mask[:lengths_y[idx]+1,idx] = 1.
        z[:lengths_z[idx],idx] = s_z
        z_mask[:lengths_z[idx]+1,idx] = 1.

    return x, x_mask, y, y_mask, z, z_mask

def grouper(text):
    """
    Group tweets into triplets
    """
    first = text[::3]
    second = text[1::3]
    third = text[2::3]
    X = (first, second, third)
    return X

def build_dictionary(text):
    """
    Build a character dictionary
    text: list of tweets
    """
    charcount = OrderedDict()
    for cc in text:
        chars = list(cc)
        for c in chars:
            if c not in charcount:
                charcount[c] = 0
            charcount[c] += 1
    chars = charcount.keys()
    freqs = charcount.values()
    sorted_idx = numpy.argsort(freqs)[::-1]

    chardict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        chardict[chars[sidx]] = idx+2 # 0: <eos>, 1: <unk>

    return chardict, charcount