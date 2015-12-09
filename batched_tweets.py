import numpy
import copy
import cPickle as pkl
from collections import OrderedDict
from settings import MAX_LENGTH, N_CHAR, MIN_LEV_DIST, MAX_TRIPLES_PER_HASHTAG, MAX_WORD_LENGTH, MAX_SEQ_LENGTH, ATTEMPTS
import json
import itertools
import random
import io
import distance

class BatchedTweets():

    def __init__(self, data, batch_size=128, maxlen=None):
        self.batch_size = 128

        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.prepare()
        self.reset()

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

def prepare_data_c2w2s(seqs_x, seqs_y, seqs_z, chardict, maxwordlen=MAX_WORD_LENGTH, maxseqlen=MAX_SEQ_LENGTH, n_chars=N_CHAR):
    """
    Put the data into format useable by the model
    """
    n_samples = len(seqs_x)
    x = numpy.zeros((n_samples,MAX_SEQ_LENGTH,MAX_WORD_LENGTH)).astype('int32')
    y = numpy.zeros((n_samples,MAX_SEQ_LENGTH,MAX_WORD_LENGTH)).astype('int32')
    z = numpy.zeros((n_samples,MAX_SEQ_LENGTH,MAX_WORD_LENGTH)).astype('int32')
    x_mask = numpy.zeros((n_samples,MAX_SEQ_LENGTH,MAX_WORD_LENGTH)).astype('float32')
    y_mask = numpy.zeros((n_samples,MAX_SEQ_LENGTH,MAX_WORD_LENGTH)).astype('float32')
    z_mask = numpy.zeros((n_samples,MAX_SEQ_LENGTH,MAX_WORD_LENGTH)).astype('float32')

    # Split words and replace by indices
    for seq_id, cc in enumerate(seqs_x):
        words = cc.split()
        for word_id, word in enumerate(words):
            if word_id >= MAX_SEQ_LENGTH:
                break
            c_len = min(MAX_WORD_LENGTH, len(word))
            x[seq_id,word_id,:c_len] = [chardict[c] if c in chardict and chardict[c] < n_chars else 0 for c in list(word)[:c_len]]
            x_mask[seq_id,word_id,:c_len] = 1.
    for seq_id, cc in enumerate(seqs_y):
        words = cc.split()
        for word_id, word in enumerate(words):
            if word_id >= MAX_SEQ_LENGTH:
                break
            c_len = min(MAX_WORD_LENGTH, len(word))
            y[seq_id,word_id,:c_len] = [chardict[c] if c in chardict and chardict[c] < n_chars else 0 for c in list(word)[:c_len]]
            y_mask[seq_id,word_id,:c_len] = 1.
    for seq_id, cc in enumerate(seqs_z):
        words = cc.split()
        for word_id, word in enumerate(words):
            if word_id >= MAX_SEQ_LENGTH:
                break
            c_len = min(MAX_WORD_LENGTH, len(word))
            z[seq_id,word_id,:c_len] = [chardict[c] if c in chardict and chardict[c] < n_chars else 0 for c in list(word)[:c_len]]
            z_mask[seq_id,word_id,:c_len] = 1.

    return numpy.expand_dims(x,axis=3), x_mask, numpy.expand_dims(y,axis=3), y_mask, numpy.expand_dims(z,axis=3), z_mask

def prepare_data(seqs_x, seqs_y, seqs_z, chardict, maxlen=MAX_LENGTH, n_chars=N_CHAR):
    """
    Put the data into format useable by the model
    """
    seqsX = []
    seqsY = []
    seqsZ = []
    for cc in seqs_x:
        seqsX.append([chardict[c] if c in chardict and chardict[c] < n_chars else 0 for c in list(cc)])
    for cc in seqs_y:
        seqsY.append([chardict[c] if c in chardict and chardict[c] < n_chars else 0 for c in list(cc)])
    for cc in seqs_z:
        seqsZ.append([chardict[c] if c in chardict and chardict[c] < n_chars else 0 for c in list(cc)])
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
    def random_combination(iterable, r):
        "Random selection from itertools.combinations(iterable, r)"
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(random.sample(xrange(n), r))
        return tuple(pool[i] for i in indices)
    tags = []
    first = []
    second = []
    with io.open(data_path,'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            for i in range(MAX_TRIPLES_PER_HASHTAG): 
                pair = random_combination(j[1],2)
                # tags is a list of meta data for each pair: [<hashtag>, <tweet 1 id>, <tweet 2 id>]
                tags.append((j[0], pair[0][0], pair[1][0]))
                first.append(pair[0][1])
                second.append(pair[1][1])

    return (first, second, tags)

def create_pairs_old(data_path):
    tags = []
    first = []
    second = []
    with io.open(data_path,'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            num_pairs = 0
            for pair in itertools.combinations(j[1],2):
                # tags is a list of meta data for each pair: [<hashtag>, <tweet 1 id>, <tweet 2 id>]
                tags.append((j[0], pair[0][0], pair[1][0]))
                first.append(pair[0][1])
                second.append(pair[1][1])
                num_pairs = num_pairs+1
                if num_pairs == MAX_TRIPLES_PER_HASHTAG:
                    break

    return (first, second, tags)

def create_pairs_old(data_path):
    tags = []
    first = []
    second = []
    with io.open(data_path,'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            num_pairs = 0
            for pair in itertools.random_combinations(j[1],2):
                # tags is a list of meta data for each pair: [<hashtag>, <tweet 1 id>, <tweet 2 id>]
                tags.append((j[0], pair[0][0], pair[1][0]))
                first.append(pair[0][1])
                second.append(pair[1][1])
                num_pairs = num_pairs+1
                if num_pairs == MAX_TRIPLES_PER_HASHTAG:
                    break

    return (first, second, tags)

def create_fewer_pairs(data_path):
    tags = []
    first = []
    second = []
    with io.open(data_path,'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            
            # keep track of pairs already generated from this hashtag
            previous_pairs = {}

            for first_tweet in j[1]:
                universe = list(j[1])
                universe.remove(first_tweet)

                # remove already-seen pairs from universe
                for key, value in previous_pairs.iteritems():
                    if value == first_tweet[0]:
                        universe = [match for match in universe if match[0] != key]
                
                # randomly pick second tweet from universe
                if (universe):
                    second_tweet = random.choice(universe)
                    previous_pairs[first_tweet[0]] = second_tweet[0]

                # tags is a list of meta data for each pair: [<hashtag>, <tweet 1 id>, <tweet 2 id>]
                tags.append((j[0], first_tweet[0], second_tweet[0]))
                first.append(first_tweet[1])
                second.append(second_tweet[1])

    return (first, second, tags)

def assign_third_old(first, second, tags):
    third = []
    valid = []

    # generate dict of <tweets: (tweet text, [hashtag list])>
    tweet_dict = {}
    for i, tag in enumerate(tags):

        # tag is a list of meta data for each pair: [<hashtag>, <tweet 1 id>, <tweet 2 id>]
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
                    if distance.levenshtein(orig_tag, new_tag) < MIN_LEV_DIST:
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

def assign_third(first, second, tags):
    first_out = []
    second_out = []
    third_out = []
    
    B = len(first)
    attempts = min(B-1,ATTEMPTS)
    for i in range(B): 
        ti = first[i]
        si = second[i]
        hi, tidi, sidi = tags[i]
        flag = attempts
        checked = []
        while (flag): 
            j = random.randrange(B)
            if j in checked:
                continue
            checked.append(j)
            flag -= 1
            hj = tags[j][0]
            if distance.levenshtein(hi, hj) > MIN_LEV_DIST: 
                if (random.getrandbits(1)): 
                    tj = first[j] 
                    newdi = tags[j][1]
                else: 
                    tj = second[j] 
                    newdi = tags[j][2]
                if (newdi != tidi ) & (newdi != sidi): 
                    first_out.append(ti)
                    second_out.append(si)
                    third_out.append(tj)
                    flag = 0
                    
    return (first_out, second_out, third_out)
