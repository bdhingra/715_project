import theano
import theano.tensor as T
import sys
import glob
import batched_tweets
import numpy as np
import cPickle as pkl
import lasagne

from settings import MAX_LENGTH, M, MAX_WORD_LENGTH, MAX_SEQ_LENGTH
from model import load_params, char2word2vec

def tnorm(tens):
    '''
    Tensor Norm
    '''
    return T.sqrt(T.sum(T.sqr(tens),axis=1))

def main(data_path, model_path):

    print("Loading data...")
    with open(data_path,'r') as f:
        valX = pkl.load(f)

    print("Preparing data...")
    val_iter = batched_tweets.BatchedTweets(valX, batch_size=512, maxlen=MAX_LENGTH)

    print("Loading dictionary...")
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    n_char = len(chardict.keys()) + 1

    # check for model files
    files = sorted(glob.glob(model_path+'model_*.npz'))
    print("Found {} model files".format(len(files)))

    for modelf in files:
        print("Computing validation cost on {}".format(modelf))

        print("Loading params...")
        params = load_params(modelf)

        print("Building network...")

        # Tweet variables
        tweet = T.itensor4()
        ptweet = T.itensor4()
        ntweet = T.itensor4()

        # masks
        t_mask = T.ftensor3()
        tp_mask = T.ftensor3()
        tn_mask = T.ftensor3()

        # Embeddings
        emb_t = char2word2vec(tweet, t_mask, params, n_char)[0]
        emb_tp = char2word2vec(ptweet, tp_mask, params, n_char)[0]
        emb_tn = char2word2vec(ntweet, tn_mask, params, n_char)[0]
        
        # batch cost
        D1 = 1 - T.batched_dot(emb_t, emb_tp)/(tnorm(emb_t)*tnorm(emb_tp))
        D2 = 1 - T.batched_dot(emb_t, emb_tn)/(tnorm(emb_t)*tnorm(emb_tn))
        gap = D1-D2+M
        loss = gap*(gap>0)
        cost = T.mean(loss)

        # Theano function
        print("Compiling theano function...")
        inps = [tweet,t_mask,ptweet,tp_mask,ntweet,tn_mask]
        cost_val = theano.function(inps,cost)

        print("Testing...")
        uidx = 0
        try:
            validation_cost = 0.
            n_val_samples = 0
            for x,y,z in val_iter:
                if not x:
                    print("Validation: Minibatch with no valid triples")
                    continue

                n_val_samples += len(x)
                x, x_m, y, y_m, z, z_m = batched_tweets.prepare_data_c2w2s(x, y, z, chardict, maxwordlen=MAX_WORD_LENGTH, maxseqlen=MAX_SEQ_LENGTH, n_chars=n_char)

                if x==None:
                    print("Validation: Minibatch with zero samples under maxlength")
                    continue

                curr_cost = cost_val(x,x_m,y,y_m,z,z_m)
                validation_cost += curr_cost*len(x)

            print("Model {} Validation Cost {}".format(modelf, validation_cost/n_val_samples))
            print("Seen {} samples.".format(n_val_samples))

        except KeyboardInterrupt:
            pass

if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
