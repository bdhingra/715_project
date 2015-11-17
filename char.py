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
import time
import cPickle as pkl

from collections import OrderedDict
from settings import NUM_EPOCHS, N_BATCH, MAX_LENGTH, N_CHAR, CHAR_DIM, SCALE, C2W_HDIM, WDIM, M, LEARNING_RATE, DISPF, SAVEF, N_VAL
from model import tweet2vec, init_params

def tnorm(tens):
    '''
    Tensor Norm
    '''
    return T.sqrt(T.sum(T.sqr(tens),axis=1))

def main(data_path,save_path,num_epochs=NUM_EPOCHS):

    print("Preparing Data...")

    # Training data
    with open(data_path,'r') as f:
	X = f.read().splitlines()

    # Build dictionary
    chardict, charcount = batched_tweets.build_dictionary(X)
    batched_tweets.save_dictionary(chardict,charcount,'%s/dict.pkl' % save_path)
    trainX = batched_tweets.grouper(X)
    train_iter = batched_tweets.BatchedTweets(trainX, validation_size=N_VAL, batch_size=N_BATCH, maxlen=MAX_LENGTH)

    # Validation set
    t_val, tp_val, tn_val = train_iter.validation_set()
    t_val, t_val_m, tp_val, tp_val_m, tn_val, tn_val_m = batched_tweets.prepare_data(t_val, tp_val, tn_val, chardict, maxlen=MAX_LENGTH)

    print("Building network...")

    # params
    n_char = len(chardict.keys()) + 1
    params = init_params(n_chars=n_char)

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

		x, x_m, y, y_m, z, z_m = batched_tweets.prepare_data(x, y, z, chardict, maxlen=MAX_LENGTH)

		if x==None:
		    print("Minibatch with zero samples under maxlength.")
		    uidx -= 1
		    continue

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
		    saveparams = OrderedDict()
		    for kk,vv in params.iteritems():
			saveparams[kk] = vv.get_value()
		    np.savez('%s/model.npz' % save_path,**saveparams)
		    print("Done.")

	    validation_cost = cost_val(t_val,t_val_m,tp_val,tp_val_m,tn_val,tn_val_m)
	    print("Epoch {} Validation Cost {}".format(epoch, validation_cost))
	    print("Seen {} samples.".format(n_samples))

    except KeyboardInterrupt:
	pass

    # Test
    #print("Testing...")
    #s = np.array([[[1],[2],[3],[4]],[[1],[1],[1],[1]]],dtype=np.int32)
    #b = np.array([[[4],[3],[2],[1]],[[1],[1],[1],[1]]],dtype=np.int32)
    #a = np.array([[[0],[2],[4],[4]],[[0],[0],[0],[0]]],dtype=np.int32)
    #s_m = np.array([[0,1,0,1],[1,1,1,0]],dtype=np.float32)
    #b_m = np.array([[0,0,0,0],[1,1,0,0]],dtype=np.float32)
    #a_m = np.array([[0,1,1,0],[1,1,1,1]],dtype=np.float32)
    #print "Input - "
    #print s
    #print b
    #print a
    #emb = t2v(s,s_m,b,b_m,a,a_m)
    #d = dist(s,s_m,b,b_m,a,a_m)
    #ll = l(s,s_m,b,b_m,a,a_m)
    #c2 = cost_val(s,s_m,b,b_m,a,a_m)
    #c1 = train(s,s_m,b,b_m,a,a_m)
    #print "Output - "
    #print "Embeddings"
    #print str(emb[0])
    #print str(emb[1])
    #print str(emb[2])
    #print "Distances"
    #print str(d[0])
    #print str(d[1])
    #print "Loss"
    #print str(ll)
    #print "Params"
    #print "source - "
    #print params
    #print "Training cost - "
    #print str(c1)
    #print "Cost function cost - "
    #print str(c2)
    #print "updates - "
    #print str(updates)

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])
