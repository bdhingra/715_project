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
import shutil

from collections import OrderedDict
from settings import NUM_EPOCHS, N_BATCH, MAX_LENGTH, N_CHAR, CHAR_DIM, SCALE, C2W_HDIM, WDIM, M, LEARNING_RATE, DISPF, SAVEF, DEBUG, REGULARIZATION, RELOAD_MODEL, RELOAD_DATA, MOMENTUM, USE_SCHEDULE
from model import tweet2vec, init_params, load_params_shared

def tnorm(tens):
    '''
    Tensor Norm
    '''
    return T.sqrt(T.sum(T.sqr(tens),axis=1))

def print_params(params):
    for kk,vv in params.iteritems():
        print("Param {} = {}".format(kk, vv.get_value()))

def display_actv(x, x_m, y, y_m, z, z_m, inps, net, prefix):
    print("\nactivations...")

    layers = lasagne.layers.get_all_layers(net)

    for l in layers:
        f = theano.function(inps, lasagne.layers.get_output(l),on_unused_input='warn')
        print("layer "+prefix+" {} - {}".format(l.name, f(x,x_m,y,y_m,z,z_m)))

def main(train_path,val_path,save_path,num_epochs=NUM_EPOCHS):

    shutil.copyfile('settings.py','%s/settings.txt'%save_path)

    print("Preparing Data...")

    # Training data
    if not RELOAD_DATA:
        print("Creating Pairs...")
        trainX = batched_tweets.create_pairs(train_path)
        valX = batched_tweets.create_pairs(val_path)
        print("Number of training pairs = {}".format(len(trainX[0])))
        print("Number of validation pairs = {}".format(len(valX[0])))
        with open('%s/train_pairs.pkl'%(save_path),'w') as f:
            pkl.dump(trainX, f)
        with open('%s/val_pairs.pkl'%(save_path),'w') as f:
            pkl.dump(valX, f)
    else:
        print("Loading Pairs...")
        with open(train_path,'r') as f:
            trainX = pkl.load(f)
        with open(val_path,'r') as f:
            valX = pkl.load(f)

    if not RELOAD_MODEL:
        # Build dictionary
        chardict, charcount = batched_tweets.build_dictionary(trainX[0] + trainX[1])
        n_char = len(chardict.keys()) + 1
        batched_tweets.save_dictionary(chardict,charcount,'%s/dict.pkl' % save_path)

        # params
        n_char = len(chardict.keys()) + 1
        params = init_params(n_chars=n_char)

    else:
        print("Loading model params...")
        params = load_params_shared('%s/model.npz' % save_path)

        print("Loading dictionary...")
        with open('%s/dict.pkl' % save_path, 'rb') as f:
            chardict = pkl.load(f)
        n_char = len(chardict.keys()) + 1

    train_iter = batched_tweets.BatchedTweets(trainX, batch_size=N_BATCH, maxlen=MAX_LENGTH)
    val_iter = batched_tweets.BatchedTweets(valX, batch_size=512, maxlen=MAX_LENGTH)

    print("Building network...")

    # Tweet variables
    tweet = T.itensor3()
    ptweet = T.itensor3()
    ntweet = T.itensor3()

    # masks
    t_mask = T.fmatrix()
    tp_mask = T.fmatrix()
    tn_mask = T.fmatrix()

    # Embeddings
    emb_t, net = tweet2vec(tweet, t_mask, params, n_char)
    emb_tp, net = tweet2vec(ptweet, tp_mask, params, n_char)
    emb_tn, net = tweet2vec(ntweet, tn_mask, params, n_char)

    # batch loss
    D1 = 1 - T.batched_dot(emb_t, emb_tp)/(tnorm(emb_t)*tnorm(emb_tp))
    D2 = 1 - T.batched_dot(emb_t, emb_tn)/(tnorm(emb_t)*tnorm(emb_tn))
    gap = D1-D2+M
    loss = gap*(gap>0)
    cost = T.mean(loss)
    cost_only = T.mean(loss)
 
    # params and updates
    print("Computing updates...")
    lr = LEARNING_RATE
    mu = MOMENTUM
    updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net, trainable=True), lr, momentum=mu)

    # Theano function
    print("Compiling theano functions...")
    inps = [tweet,t_mask,ptweet,tp_mask,ntweet,tn_mask]
    #dist = theano.function(inps,[D1,D2])
    #l = theano.function(inps,loss)
    #t2v = theano.function(inps,[emb_t,emb_tp,emb_tn])
    cost_val = theano.function(inps,[cost_only, emb_t, emb_tp, emb_tn])
    train = theano.function(inps,cost,updates=updates)

    # Training
    print("Training...")
    uidx = 0
    try:
        for epoch in range(num_epochs):
            n_samples = 0
            train_cost = 0.
            print("Epoch {}".format(epoch))

            if USE_SCHEDULE:
                # schedule
                if epoch > 0 and (epoch+1) % 10 == 0:
                    print("Updating Schedule...")
                    lr = max(1e-5,lr/10)
                    mu = mu - 0.1
                    updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net, trainable=True), lr, momentum=mu)
                    train = theano.function(inps,cost,updates=updates)

            if epoch >= 10:
                cost = T.mean(loss) + REGULARIZATION*lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2)
                reg_only = REGULARIZATION*lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2)
                reg_val = theano.function([],reg_only)
                train = theano.function(inps,cost,updates=updates)

            ud_start = time.time()
            for x,y,z in train_iter:
                if not x:
                    print("Minibatch with no valid triples")
                    continue

                n_samples +=len(x)
                uidx += 1

                if DEBUG and uidx > 3:
                    sys.exit()

                if DEBUG:
                    print("Tweets = {}".format(x[:5]))

                x, x_m, y, y_m, z, z_m = batched_tweets.prepare_data(x, y, z, chardict, maxlen=MAX_LENGTH, n_chars=n_char)

                if x==None:
                    print("Minibatch with zero samples under maxlength.")
                    uidx -= 1
                    continue

                if DEBUG:
                    print("Params before update...")
                    print_params(params)
                    display_actv(x,x_m,y,y_m,z,z_m,inps,net,'before')
                    cb, embb, embb_p, embb_n = cost_val(x,x_m,y,y_m,z,z_m)

                curr_cost = train(x,x_m,y,y_m,z,z_m)
                ud = time.time() - ud_start
                train_cost += curr_cost*len(x)

                if DEBUG:
                    print("Params after update...")
                    print_params(params)
                    display_actv(x,x_m,y,y_m,z,z_m,inps,net,'after')
                    ca, emba, emba_p, emba_n = cost_val(x,x_m,y,y_m,z,z_m)
                    print("Embeddings before = {}".format(embb[:5]))
                    print("Embeddings after = {}".format(emba[:5]))
                    print("Cost before update = {} \nCost after update = {}".format(cb, ca))

                if np.isnan(curr_cost) or np.isinf(curr_cost):
                    print("Nan detected.")
                    return

                if np.mod(uidx, DISPF) == 0:
                    print("Epoch {} Update {} Cost {} Time {} Samples {}".format(epoch,uidx,curr_cost,ud,len(x)))

                if np.mod(uidx,SAVEF) == 0:
                    print("Saving...")
                    saveparams = OrderedDict()
                    for kk,vv in params.iteritems():
                        saveparams[kk] = vv.get_value()
                        np.savez('%s/model.npz' % save_path,**saveparams)
                    print("Done.")

            print("Computing Validation Cost...")
            validation_cost = 0.
            n_val_samples = 0
            for x,y,z in val_iter:
                if not x:
                    print("Validation: Minibatch with no valid triples")
                    continue

                n_val_samples += len(x)
                x, x_m, y, y_m, z, z_m = batched_tweets.prepare_data(x, y, z, chardict, maxlen=MAX_LENGTH, n_chars=n_char)

                if x==None:
                    print("Validation: Minibatch with zero samples under maxlength")
                    continue

                curr_cost, _, _, _ = cost_val(x,x_m,y,y_m,z,z_m)
                validation_cost += curr_cost*len(x)

            if epoch >= 10:
                regularization_cost = reg_val()
            else:
                regularization_cost = 0
            print("Epoch {} Training Cost {} Validation Cost {} Regularization Cost {}".format(epoch, train_cost/n_samples, validation_cost/n_val_samples, regularization_cost))
            print("Seen {} samples.".format(n_samples))

            for kk,vv in params.iteritems():
                print("Param {} Epoch {} Max {} Min {}".format(kk, epoch, np.max(vv.get_value()), np.min(vv.get_value())))

            print("Saving...")
            saveparams = OrderedDict()
            for kk,vv in params.iteritems():
                saveparams[kk] = vv.get_value()
                np.savez('%s/model_%d.npz' % (save_path,epoch),**saveparams)
            print("Done.")
            
            if False:
                # store embeddings and data
                features = np.zeros((len(train_iter.data[0]),3*WDIM))
                distances = np.zeros((len(train_iter.data[0]),2))
                for idx, triple in enumerate(zip(train_iter.data[0],train_iter.data[1],train_iter.data[2])):
                    x, x_m, y, y_m, z, z_m = batched_tweets.prepare_data([triple[0]], [triple[1]], [triple[2]], chardict, maxlen=MAX_LENGTH, n_chars=n_char)
                    if x==None:
                        continue
                    emb1, emb2, emb3 = t2v(x,x_m,y,y_m,z,z_m)
                    emb1 = np.reshape(emb1, (WDIM))
                    emb2 = np.reshape(emb2, (WDIM))
                    emb3 = np.reshape(emb3, (WDIM))
                    features[idx,:] = np.concatenate((emb1,emb2,emb3),axis=0)
                    distances[idx,0] = 1-np.dot(emb1,emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
                    distances[idx,1] = 1-np.dot(emb1,emb3)/(np.linalg.norm(emb1)*np.linalg.norm(emb3))
                with open('debug/feat_%d.npy'%epoch,'w') as df:
                    np.save(df,features)
                with open('debug/dist_%d.npy'%epoch,'w') as ds:
                    np.save(ds,distances)
        if False:
            with open('debug/data.txt','w') as dd:
                for triple in zip(train_iter.data[0],train_iter.data[1],train_iter.data[2]):
                    dd.write('%s\t%s\t%s\n' % (triple[0],triple[1],triple[2]))

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])
