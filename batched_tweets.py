import numpy
import copy
import cPickle as pkl
from collections import OrderedDict
from settings import MAX_LENGTH, N_CHAR
import json
import itertools
import random
import io
import distance

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
        self.first = self.data[0]
        self.second = self.data[1]
        self.tags = self.data[2]

        # find the unique lengths
        self.lengths = [len(list(cc)) for cc in self.first]
        self.len_unique = numpy.unique(self.lengths)
        # remove any overly long sentences
        if self.maxlen:
            self.len_unique = [ll for ll in self.len_unique if ll <= self.maxlen]

        # indices of unique lengths
        self.len_indices = dict()
        self.len_counts = dict()
        self.total = 0
        for ll in self.len_unique:
            self.len_indices[ll] = numpy.where(self.lengths == ll)[0]
            self.len_counts[ll] = len(self.len_indices[ll])
            self.total += len(self.len_indices[ll])

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

        first = [self.first[ii] for ii in curr_indices]
        second = [self.second[ii] for ii in curr_indices]
        tags = [self.tags[ii] for ii in curr_indices]
        (first, second, third) = assign_third(first, second, tags)

        return first, second, third

    def __iter__(self):
        return self

def prepare_data(seqs_x, seqs_y, seqs_z, chardict, maxlen=MAX_LENGTH, n_chars=N_CHAR):
    """
    Put the data into format useable by the model
    """
    seqsX = []
    seqsY = []
    seqsZ = []
    for cc in seqs_x:
        seqsX.append([chardict[c] if chardict[c] < n_chars else 0 for c in list(cc)])
    for cc in seqs_y:
        seqsY.append([chardict[c] if chardict[c] < n_chars else 0 for c in list(cc)])
    for cc in seqs_z:
        seqsZ.append([chardict[c] if chardict[c] < n_chars else 0 for c in list(cc)])
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

    x = numpy.zeros((n_samples,MAX_LENGTH)).astype('int32')
    y = numpy.zeros((n_samples,MAX_LENGTH)).astype('int32')
    z = numpy.zeros((n_samples,MAX_LENGTH)).astype('int32')
    x_mask = numpy.zeros((n_samples,MAX_LENGTH)).astype('float32')
    y_mask = numpy.zeros((n_samples,MAX_LENGTH)).astype('float32')
    z_mask = numpy.zeros((n_samples,MAX_LENGTH)).astype('float32')
    for idx, [s_x, s_y, s_z] in enumerate(zip(seqs_x,seqs_y,seqs_z)):
        x[idx,:lengths_x[idx]] = s_x
        x_mask[idx,:lengths_x[idx]] = 1.
        y[idx,:lengths_y[idx]] = s_y
        y_mask[idx,:lengths_y[idx]] = 1.
        z[idx,:lengths_z[idx]] = s_z
        z_mask[idx,:lengths_z[idx]] = 1.

    return numpy.expand_dims(x,axis=2), x_mask, numpy.expand_dims(y,axis=2), y_mask, numpy.expand_dims(z,axis=2), z_mask

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
        chardict[chars[sidx]] = idx + 1

    return chardict, charcount

def save_dictionary(worddict, wordcount, loc):
    """
    Save a dictionary to the specified location 
    """
    with open(loc, 'wb') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)

def create_pairs(data_path):
    tags = []
    first = []
    second = []
    with open(data_path,'r') as f:
        for line in f.readlines():
            j = json.loads(line)
            if j[0] != "" and len(j[1]) > 1:
                for pair in itertools.combinations(j[1],2):
                    tags.append((j[0], pair[0][0], pair[1][0]))
                    first.append(pair[0][1])
                    second.append(pair[1][1])

    return (first, second, tags)

def assign_third(first, second, tags):
    third = []
    valid = []

    # generate dict of <tweets: (tweet text, [hashtag list])>
    tweet_dict = {}
    for i, tag in enumerate(tags):
        tweet = tag[1]   
        if not tweet in tweet_dict:
            tweet_dict[tweet] = (first[i],[tag[0]])
        else:
            if tag[0] not in tweet_dict[tweet][1]:
                tweet_dict[tweet][1].append(tag[0])
        
        tweet = tag[2]
        if not tweet in tweet_dict:
            tweet_dict[tweet] = (second[i],[tag[0]])
        else:
            if tag[0] not in tweet_dict[tweet][1]:
                tweet_dict[tweet][1].append(tag[0])

    # create universe of valid third tweets and randomly sample
    for i, tag in enumerate(tags):
        universe = []
        first_id = tag[1]
        second_id = tag[2]

        # create combined list of hashtags from first & second tweets
        all_tags = tweet_dict[first_id][1]+tweet_dict[second_id][1]
        
        # check all tweets in batch for validity
        for tweet in tweet_dict:
            similar = False
            for orig_tag in set(all_tags):
                for new_tag in set(tweet_dict[tweet][1]):

                    # if levenshtein distance is too small between any hashtags,
                    # third tweet is not valid
                    if distance.levenshtein(orig_tag, new_tag) < 5:
                        similar = True
                        break

            if not similar:
                universe.append(tweet_dict[tweet][0])
        
        # if there are any valid third tweets, randomly choose one
        if universe:
            third.append(random.choice(universe))
            valid.append(True)
        else:
            third.append("")
            valid.append(False)

    # return only pairs where a valid third tweet was found
    first_out = []
    second_out = []
    third_out = []
    for i, check in enumerate(valid):
        if check:
            first_out.append(first[i])
            second_out.append(second[i])
            third_out.append(third[i])

    return (first_out, second_out, third_out)