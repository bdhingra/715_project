import numpy as np
import lasagne
import theano
import theano.tensor as T
import sys
import batched_tweets
import cPickle as pkl

from settings import NUM_EPOCHS, N_BATCH, MAX_LENGTH, N_CHAR, CHAR_DIM, SCALE, C2W_HDIM, WDIM, M, LEARNING_RATE, DISPF, SAVEF, N_VAL, DEBUG, REGULARIZATION, RELOAD_MODEL, RELOAD_DATA, MAX_WORD_LENGTH, MAX_SEQ_LENGTH
from model import char2word2vec, init_params_c2w2s, load_params_shared

data_path = 'model/sample_190M/c2w2s_small/train_pairs.pkl'

def tnorm(tens):
    '''
    Tensor Norm
    '''
    return T.sqrt(T.sum(T.sqr(tens),axis=1))

# load data
print("loading data...")
trainX = pkl.load(open(data_path,'r'))

# dictionary
chardict, charcount = batched_tweets.build_dictionary(trainX[0] + trainX[1])
n_char = len(chardict.keys()) + 1

# model params
params = init_params_c2w2s(n_chars=n_char)

# batches
print("preparing batches...")
train_iter = batched_tweets.BatchedTweets(trainX, batch_size=N_BATCH, maxlen=MAX_LENGTH)

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

# batch loss
D1 = 1 - T.batched_dot(emb_t, emb_tp)/(tnorm(emb_t)*tnorm(emb_tp))
D2 = 1 - T.batched_dot(emb_t, emb_tn)/(tnorm(emb_t)*tnorm(emb_tn))
gap = D1-D2+M
loss = gap*(gap>0)
cost = T.mean(loss)

# Theano function
print("Compiling theano functions...")
inps = [tweet,t_mask,ptweet,tp_mask,ntweet,tn_mask]
dist = theano.function(inps,[D1,D2])
l = theano.function(inps,loss)
t2v = theano.function(inps,[emb_t,emb_tp,emb_tn])

#inps = [tweet,t_mask]
#l = theano.function([tweet],lookup)
#fg = theano.function(inps,fgru)
#bg = theano.function(inps,bgru)
#f = theano.function(inps,ff)
#b = theano.function(inps,bb)
#fde = theano.function(inps,fd)
#bde = theano.function(inps,bd)
#c = theano.function(inps,c2w)

# test
print("testing...")
xt = []
while not xt:
    xt,yt,zt = train_iter.next()
#xt=['dey *&wubp']
#yt=['hye bye']
#zt=['bye bye']
x, x_m, y, y_m, z, z_m = batched_tweets.prepare_data_c2w2s(xt, yt, zt, chardict, maxwordlen=MAX_WORD_LENGTH, maxseqlen=MAX_SEQ_LENGTH, n_chars=n_char)
print(u"triple - {} \t {} \t {}".format(xt[:2],yt[:2],zt[:2]))

#fns = []
#layers = lasagne.layers.get_all_layers(net)
#inps = [tweet,t_mask]
#for l in layers:
#    f = theano.function(inps,lasagne.layers.get_output(l),on_unused_input='warn')
#l = layers[-1]
#f = theano.function(inps,lasagne.layers.get_output(l))
#print("At layer {} - Input = {} Output = {}".format(l.name, xt, f(x,x_m)))
#print("At layer {} - Input = {} Output = {}".format(l.name, zt, f(y,y_m)))
#print("At layer {} - Input = {} Output = {}".format(l.name, zt, f(z,z_m)))

ex,ey,ez = t2v(x,x_m,y,y_m,z,z_m)
d1,d2 = dist(x,x_m,y,y_m,z,z_m)
lo = l(x,x_m,y,y_m,z,z_m)
print("embedding 1 = {}".format(ex[:2]))
print("embedding 2 = {}".format(ey[:2]))
print("embedding 3 = {}".format(ez[:2]))
print("d1 = {}, d2 = {}".format(d1[:2],d2[:2]))
print("loss = {}".format(lo[:2]))
#print("x = {}".format(x))
#xl = l(x)
#xfg = fg(x,x_m)
#xbg = bg(x,x_m)
#xf = f(x,x_m)
#xb = b(x,x_m)
#xfde = fde(x,x_m)
#xbde = bde(x,x_m)
#xc = c(x,x_m)
#print("lookup = {}".format(xl))
#print("forward gru = {}".format(xfg))
#print("backward gru = {}".format(xbg))
#print("slice forward = {}".format(xf))
#print("slice backward = {}".format(xb))
#print("forward dense = {}".format(xfde))
#print("backward dense = {}".format(xbde))
#print("embedding = {}".format(xc))
