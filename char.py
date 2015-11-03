'''
Tweet2Vec trainer
'''

import numpy as np
import lasagne
import theano
import theano.tensor as T
import random
import sys
import batched_tweets

from collections import OrderedDict

# Number of epochs
NUM_EPOCHS = 1
# Batch size
N_BATCH = 2
# Max sequence length
MAX_LENGTH = 4
# Number of unique characters
N_CHAR = 5
# Dimensionality of character lookup
CHAR_DIM = 2
# Initialization scale
SCALE = 0.1
# Dimensionality of C2W hidden states
C2W_HDIM = 4
# Dimensionality of word vectors
WDIM = 3
# Gap parameter
M = 0.5
# Learning rate
LEARNING_RATE = .001
# Display frequency
DISPF = 1
# Save frequency
SAVEF = 10
# Validation set
N_VAL = 1000

def tnorm(tens):
    '''
    Tensor Norm
    '''
    return T.sqrt(T.sum(T.sqr(tens),axis=1))

def tweet2vec(tweet,mask,params):
    '''
    Tweet2Vec
    '''
    # Input layer over characters
    l_in_source = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_LENGTH,1), input_var=tweet, name='input')

    # Mask layer for variable length sequences
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH,MAX_LENGTH), input_var=mask, name='mask')

    # lookup
    l_clookup_source = lasagne.layers.EmbeddingLayer(l_in_source, input_size=N_CHAR, output_size=CHAR_DIM, W=params['Wc'])

    # f-GRU
    c2w_f_reset = lasagne.layers.Gate(W_in=params['W_c2w_f_r'], W_hid=params['U_c2w_f_r'], W_cell=None, b=params['b_c2w_f_r'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_f_update = lasagne.layers.Gate(W_in=params['W_c2w_f_z'], W_hid=params['U_c2w_f_z'], W_cell=None, b=params['b_c2w_f_z'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_f_hidden = lasagne.layers.Gate(W_in=params['W_c2w_f_h'], W_hid=params['U_c2w_f_h'], W_cell=None, b=params['b_c2w_f_h'], nonlinearity=lasagne.nonlinearities.tanh)

    l_fgru_source = lasagne.layers.GRULayer(l_clookup_source, C2W_HDIM, resetgate=c2w_f_reset, updategate=c2w_f_update, hidden_update=c2w_f_hidden, hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=l_mask)

    # b-GRU
    c2w_b_reset = lasagne.layers.Gate(W_in=params['W_c2w_b_r'], W_hid=params['U_c2w_b_r'], W_cell=None, b=params['b_c2w_b_r'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_b_update = lasagne.layers.Gate(W_in=params['W_c2w_b_z'], W_hid=params['U_c2w_b_z'], W_cell=None, b=params['b_c2w_b_z'], nonlinearity=lasagne.nonlinearities.sigmoid)
    c2w_b_hidden = lasagne.layers.Gate(W_in=params['W_c2w_b_h'], W_hid=params['U_c2w_b_h'], W_cell=None, b=params['b_c2w_b_h'], nonlinearity=lasagne.nonlinearities.tanh)

    l_bgru_source = lasagne.layers.GRULayer(l_clookup_source, C2W_HDIM, resetgate=c2w_b_reset, updategate=c2w_b_update, hidden_update=c2w_b_hidden, hid_init=lasagne.init.Constant(0.), backwards=True, learn_init=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False, precompute_input=True, mask_input=l_mask)

    # Slice final states
    l_f_source = lasagne.layers.SliceLayer(l_fgru_source, -1, 1)
    l_b_source = lasagne.layers.SliceLayer(l_bgru_source, 0, 1)

    # Dense layer
    l_fdense_source = lasagne.layers.DenseLayer(l_f_source, WDIM, W=params['W_c2w_df'], b=params['b_c2w_df'], nonlinearity=None)
    l_bdense_source = lasagne.layers.DenseLayer(l_b_source, WDIM, W=params['W_c2w_db'], b=params['b_c2w_db'], nonlinearity=None)
    l_c2w_source = lasagne.layers.ElemwiseSumLayer([l_fdense_source, l_bdense_source], coeffs=1)

    return lasagne.layers.get_output(l_c2w_source)
    
def init_params():
    '''
    Initialize all params
    '''
    params = OrderedDict()

    np.random.seed(0)

    # lookup table
    params['Wc'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(N_CHAR,CHAR_DIM)).astype('float32'), name='Wc')

    # f-GRU
    params['W_c2w_f_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_f_r')
    params['W_c2w_f_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_f_z')
    params['W_c2w_f_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_f_h')
    params['b_c2w_f_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name='b_c2w_f_r')
    params['b_c2w_f_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name='b_c2w_f_z')
    params['b_c2w_f_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name='b_c2w_f_h')
    params['U_c2w_f_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_f_r')
    params['U_c2w_f_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_f_z')
    params['U_c2w_f_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_f_h')

    # b-GRU
    params['W_c2w_b_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_b_r')
    params['W_c2w_b_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_b_z')
    params['W_c2w_b_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(CHAR_DIM,C2W_HDIM)).astype('float32'), name='W_c2w_b_h')
    params['b_c2w_b_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name='b_c2w_b_r')
    params['b_c2w_b_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name='b_c2w_b_z')
    params['b_c2w_b_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM)).astype('float32'), name='b_c2w_b_h')
    params['U_c2w_b_r'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_b_r')
    params['U_c2w_b_z'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_b_z')
    params['U_c2w_b_h'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,C2W_HDIM)).astype('float32'), name='U_c2w_b_h')

    # dense
    params['W_c2w_df'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,WDIM)).astype('float32'), name='W_c2w_df')
    params['W_c2w_db'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(C2W_HDIM,WDIM)).astype('float32'), name='W_c2w_db')
    params['b_c2w_df'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(WDIM)).astype('float32'), name='b_c2w_df')
    params['b_c2w_db'] = theano.shared(np.random.uniform(low=-SCALE, high=SCALE, size=(WDIM)).astype('float32'), name='b_c2w_db')

    return params
    
def main(data_path,save_path,num_epochs=NUM_EPOCHS):

    print("Building network...")

    # params
    params = init_params()

    # Tweet variables
    tweet = T.itensor3()
    ptweet = T.itensor3()
    ntweet = T.itensor3()

    # masks
    t_mask = T.fmatrix()
    tp_mask = T.fmatrix()
    tn_mask = T.fmatrix()

    # Embeddings
    emb_t = tweet2vec(tweet, t_mask, params)
    emb_tp = tweet2vec(ptweet, tp_mask, params)
    emb_tn = tweet2vec(ntweet, tn_mask, params)

    # batch loss
    D1 = 1 - T.batched_dot(emb_t, emb_tp)/(tnorm(emb_t)*tnorm(emb_tp))
    D2 = 1 - T.batched_dot(emb_t, emb_tn)/(tnorm(emb_t)*tnorm(emb_tn))
    gap = D1-D2+M
    loss = gap*(gap>0)
    cost = T.sum(loss)

    # params and updates
    print("Computing updates...")
    updates = lasagne.updates.adagrad(cost, params.values(), LEARNING_RATE)

    # Theano function
    print("Compiling theano functions...")
    inps = [tweet,t_mask,ptweet,tp_mask,ntweet,tn_mask]
    dist = theano.function(inps,[D1,D2])
    l = theano.function(inps,loss)
    t2v = theano.function(inps,[emb_t,emb_tp,emb_tn])
    cost_val = theano.function(inps,cost)
    train = theano.function(inps,cost,updates=updates)

    # Training data
    with open(data_path,'r') as f:
	X = f.read().splitlines()

    # Build dictionary
    chardict = batched_tweets.build_dictionary(X)
    trainX = batched_tweets.grouper(X)
    train_iter = batched_tweets.BatchedTweets(trainX, validation_size=N_VAL, batch_size=N_BATCH, maxlen=MAX_LENGTH)

    # Validation set
    t_val, tp_val, tn_val = train_iter.validation_set()
    t_val, t_val_m, tp_val, tp_val_m, tn_val, tn_val_m = batched_tweets.prepare_data(t_val, tp_val, tn_val, maxlen=MAX_LENGTH)

    # Training
    print("Training...")
    uidx = 0
    try:
	for epoch in range(num_epochs):
	    n_samples = 0
	    print("Epoch {}".format(epoch))

	    for x,y,z in train_iter:
		n_samples +=len(x)
		uidx += 1

		x, x_m, y, y_m, z, z_m = batched_tweets.prepare_data(x, y, z, maxlen=MAX_LENGTH)

		if x==None:
		    print("Minibatch with zero samples under maxlength.")
		    uidx -= 1

		ud_start = time.time()
		curr_cost = train(x,x_m,y,y_m,z,z_m)
		ud = time.time() - ud_start

		if np.isnan(curr_cost) or np.isinf(curr_cost):
		    print("Nan detected.")
		    return

		if np.mod(uidx, DISPF) == 0:
		    print("Epoch {} Update {} Cost {} Time {}".format(epoch,uidx,curr_cost,ud))

		if np.mod(uidx,SAVEF) == 0:
		    print("Saving...")
		    args = params.keys()
		    kwds = [v.get_value() for v in params.values()]
		    np.savez(SAVEF,*args,**kwds)
		    print("Done.")

	    validation_cost = cost_val(t_val,t_val_m,tp_val,tp_val_m,tn_val,tn_val_m)
	    print("Epoch {} Validation Cost {}".format(epoch, validation_cost))
	    print("Seen {} samples.".format(n_samples))

    except KeyboardInterrupt:
	pass

    # Test
    print("Testing...")
    s = np.array([[[1],[2],[3],[4]],[[1],[1],[1],[1]]],dtype=np.int32)
    b = np.array([[[4],[3],[2],[1]],[[1],[1],[1],[1]]],dtype=np.int32)
    a = np.array([[[0],[2],[4],[4]],[[0],[0],[0],[0]]],dtype=np.int32)
    s_m = np.array([[0,1,0,1],[1,1,1,0]],dtype=np.float32)
    b_m = np.array([[0,0,0,0],[1,1,0,0]],dtype=np.float32)
    a_m = np.array([[0,1,1,0],[1,1,1,1]],dtype=np.float32)
    print "Input - "
    print s
    print b
    print a
    emb = t2v(s,s_m,b,b_m,a,a_m)
    d = dist(s,s_m,b,b_m,a,a_m)
    ll = l(s,s_m,b,b_m,a,a_m)
    c2 = cost_val(s,s_m,b,b_m,a,a_m)
    c1 = train(s,s_m,b,b_m,a,a_m)
    print "Output - "
    print "Embeddings"
    print str(emb[0])
    print str(emb[1])
    print str(emb[2])
    print "Distances"
    print str(d[0])
    print str(d[1])
    print "Loss"
    print str(ll)
    print "Params"
    print "source - "
    print params
    print "Training cost - "
    print str(c1)
    print "Cost function cost - "
    print str(c2)
    print "updates - "
    print str(updates)

if __name__ == '__main__':
    main(sys.argv[0],sys.argv[1],sys.argv[2])
